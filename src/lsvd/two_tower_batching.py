"""Remaining-target priority batching for the approved mean-tower diagnosis."""

from collections import defaultdict
import heapq

import numpy as np


def remaining_user_batches(chunks, chunk_users, order, batch_size, user_cap):
    user_chunks = defaultdict(list)
    for index in order:
        user_chunks[int(chunk_users[index])].append(chunks[index])
    queues = {user: np.concatenate(parts) for user, parts in user_chunks.items()}
    cursors = {user: 0 for user in queues}
    # Equal remaining counts follow the first shuffled occurrence of each user.
    pending = [(-len(queues[user]), priority, user)
               for priority, user in enumerate(queues)]
    heapq.heapify(pending)
    while pending:
        parts = []
        count = 0
        deferred = []
        while pending and count < batch_size:
            negative_remaining, priority, user = heapq.heappop(pending)
            take = min(-negative_remaining, user_cap, batch_size - count)
            low = cursors[user]
            high = low + take
            parts.append(queues[user][low:high])
            cursors[user] = high
            count += take
            if high < len(queues[user]):
                deferred.append((high - len(queues[user]), priority, user))
        # Reinsert only after building this batch, preserving the per-user cap.
        for entry in deferred:
            heapq.heappush(pending, entry)
        yield np.concatenate(parts)
