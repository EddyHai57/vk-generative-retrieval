"""Expand saved top1000 lists to every ruler/K for all thirteen diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .two_tower import Catalog, TowerConfig
from .two_tower_data import subset_evaluation
from .two_tower_retrieval import diagnostic_evaluations, official_rank_predictions
from .evaluation import EvaluationData, EvaluationResult
from .baselines import BaselineData, global_popularity


def local_popularity_predictions(data):
    baseline_data = BaselineData(
        dict(user_id=data.user_ids),
        data.items,
        data.history_user_ids,
        data.history_offsets,
        data.history_items,
    )
    return global_popularity(baseline_data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args(argv)
    run = json.loads((args.run / "summary.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    first = run["models"][0]["structure"]
    users = np.load(args.run / first / "retrieval/user_ids.npy", allow_pickle=False)
    data = subset_evaluation(EvaluationData.load(args.foundation), users)
    import duckdb

    connection = duckdb.connect()
    connection.execute("SET memory_limit='8GB'")
    from .two_tower_runtime import available_resources
    connection.execute(f"SET threads={available_resources()['cpu_threads']}")
    connection.execute("SET temp_directory=?", [str(args.output / "tmp")])
    columns = connection.execute(
        """SELECT i.item_id,i.author_id,i.duration,i.train_count,m.train_interactions_rank AS rank
     FROM read_parquet(?) i JOIN read_parquet(?) m USING(item_id) ORDER BY i.item_id""",
        [str(args.foundation / "items.parquet"), str(args.metadata)],
    ).fetchnumpy()
    connection.close()
    catalog = Catalog(columns, TowerConfig(id_min_count=run["config"]["id_min_count"]))
    records = []
    for model in run["models"]:
        structure = model["structure"]
        for path, label in [
            ("retrieval", structure),
            ("cold-retrieval", structure + "_cold"),
        ]:
            folder = args.run / structure / path
            if not np.array_equal(
                users, np.load(folder / "user_ids.npy", allow_pickle=False)
            ):
                raise ValueError(
                    "saved predictions must share exactly the same query order"
                )
            predictions = np.load(
                folder / "predictions.npy", mmap_mode="r", allow_pickle=False
            )
            print("扩展三尺子与三种K的分组评估", label, flush=True)
            _, groups = diagnostic_evaluations(
                catalog,
                data,
                predictions,
                label,
                args.output / label,
                args.replicates,
                diagnostic_ks=(50, 200, 1000),
                save_diagnostics=True,
            )
            original = model["groups"] if path == "retrieval" else model["cold_groups"]
            for before, after in zip(original, groups):
                for key in ("group", "k", "hits", "n_targets", "n_users"):
                    if before[key] != after[key]:
                        raise ValueError(f"group evaluation changed {key}: {label}")
                if before["macro_capped"] is not None:
                    np.testing.assert_allclose(
                        before["macro_capped"],
                        after["macro_capped"],
                        rtol=0,
                        atol=1e-15,
                    )
            records.append(
                dict(
                    model=label,
                    main200_matches_original=True,
                    diagnostic_groups=len(groups),
                )
            )
    for name, predictions in [
        ("official_rank", official_rank_predictions(catalog, data)),
        ("local_global_popularity", local_popularity_predictions(data)),
    ]:
        print("扩展不训练基线分组评估", name, flush=True)
        base, groups = diagnostic_evaluations(
            catalog,
            data,
            predictions,
            name,
            args.output / name,
            args.replicates,
            diagnostic_ks=(50, 200, 1000),
            save_diagnostics=True,
        )
        original = (
            run["official_rank"]
            if name == "official_rank"
            else run["global_popularity"]
        )
        selected = next(
            row
            for row in base.metrics
            if row["ruler"] == "main"
            and row["place_scope"] == "0/1"
            and row["seen_scope"] == "all"
            and row["breakdown"] == "all"
            and row["k"] == 200
        )
        for key in ("hits", "n_targets", "n_users"):
            if original[key] != selected[key]:
                raise ValueError(f"baseline changed {key}: {name}")
        original_folder = args.run / (
            "official-rank/all" if name == "official_rank" else "global-popularity"
        )
        reference = EvaluationResult.load(original_folder)
        np.testing.assert_array_equal(base.user_ids, reference.user_ids)
        np.testing.assert_array_equal(base.counts, reference.counts)
        np.testing.assert_array_equal(base.hits, reference.hits)
        records.append(
            dict(
                model=name, main200_matches_original=True, diagnostic_groups=len(groups)
            )
        )
    for name in ("last_author_popularity", "covisitation", "content_mean"):
        folder = args.foundation.parent / "predictions" / name
        source_users = np.load(folder / "user_ids.npy", allow_pickle=False)
        indices = np.searchsorted(source_users, users)
        if np.any(indices >= len(source_users)) or not np.array_equal(
            source_users[indices], users
        ):
            raise ValueError(
                "foundation baseline queries do not cover the selected users"
            )
        saved = np.load(folder / "predictions.npy", mmap_mode="r", allow_pickle=False)
        predictions = saved[indices]
        print("复用第十三轮固定预测并扩展分组", name, flush=True)
        _, groups = diagnostic_evaluations(
            catalog,
            data,
            predictions,
            name,
            args.output / name,
            args.replicates,
            diagnostic_ks=(50, 200, 1000),
            save_diagnostics=True,
        )
        records.append(
            dict(
                model=name,
                source="round13 frozen predictions",
                diagnostic_groups=len(groups),
                original_manifest_sha256=hashlib.sha256(
                    (folder / "manifest.json").read_bytes()
                ).hexdigest(),
            )
        )
    summary = dict(
        queries=len(users),
        models=records,
        ks=[50, 200, 1000],
        rulers=["main", "secondary", "like"],
        original_summary_sha256=hashlib.sha256(
            (args.run / "summary.json").read_bytes()
        ).hexdigest(),
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
