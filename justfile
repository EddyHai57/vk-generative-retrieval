# Convenience only; allow rules target vkgr-fixed directly, never arbitrary justfiles.
ops-plan:
    ./ops/bin/vkgr-fixed plan

ops-status:
    ./ops/bin/vkgr-fixed status execute

ops-resources:
    ./ops/bin/vkgr-fixed resources execute

ops-queue:
    ./ops/bin/vkgr-fixed queue execute

ops-receipts:
    ./ops/bin/vkgr-fixed receipts execute

ops-deploy-plan:
    ./ops/bin/vkgr-fixed deploy plan

ops-deploy:
    ./ops/bin/vkgr-fixed deploy execute
