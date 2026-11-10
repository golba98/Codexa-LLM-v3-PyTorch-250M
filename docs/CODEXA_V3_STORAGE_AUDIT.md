# Codexa v3 — Full 512GB Storage Audit

Audit date: 2026-11-05T14:22:14.073132+00:00. Inspection only. No cleanup, relocation, dataset rebuild, training, optimizer update or checkpoint modification was performed. The sole requested content output is this report.

## Findings

The drive reports **405.299 GiB used** and **58.281 GiB available**. Codexa v3 accounts for **146.275 GiB**; approximately **259.022 GiB** of listed allocation belongs to other projects and volume metadata. Older PyTorch workspace outputs, saved artifacts and NumPy runs are major consumers. Their age does not make them disposable.

Low-risk candidates total **3.099 GiB**. A separate, conditional set totals **25.502 GiB** and requires dependency/retention approval. Combined selected potential is **28.600 GiB**. No space has actually been reclaimed. High-risk historical state is not included in the low-risk total.

The largest regular file is an older saved training-token array at 25,639,702,528 allocated bytes (23.879 GiB). Multiple older 900M/934M checkpoints consume about 3.48–5.25 GiB each. Those retained models and large corpus/index files explain most consumption; ordinary generated caches are a small fraction.

Dataset rebuilds have created substantial extra allocations. The original reviewed snapshot and its successors are independently named, immutable artifacts. Some shards are hardlinked, while changed shards and document mappings are newly stored. It is incorrect both to call every snapshot a full extra copy and to assume they cost no space. Continuation bundles also each serialize a complete model plus optimizer state.

## Filesystem identity, capacity and accounting

Verified mount: `/mnt/codexa`; device `/dev/mapper/codexa-assets`; ext4; UUID `279352d0-0011-4733-8348-6815a4308196`. Physical SSD: `/dev/sda`, stable ID `ata-HS-SSD-E100_512G_30035028816`; partition is LUKS UUID `524123d6-12ab-4b4d-8f8d-19685ba7eeea`. Mount options include `rw,nosuid,nodev,noexec,relatime,seclabel`. No nested mounts were found.

| Measurement | Bytes | GiB |
| --- | --- | --- |
| Physical SSD | 512,110,190,592 | 476.940 |
| Partition | 512,107,741,184 | 476.938 |
| LUKS mapped block device | 512,090,963,968 | 476.922 |
| Filesystem reported capacity | 502,903,234,560 | 468.365 |
| Filesystem used | 435,186,528,256 | 405.299 |
| Free blocks, including privileged reserve | 67,716,706,304 | 63.066 |
| Available to ordinary user | 62,579,019,776 | 58.281 |
| Free but unavailable to ordinary user | 5,137,686,528 | 4.785 |
| Mapped-device size minus filesystem capacity | 9,187,729,408 | 8.557 |
| Listed inode allocation | 435,184,422,912 | 405.297 |
| Used minus listed inode allocation | 2,105,344 | 0.002 |

Block/fragment size is 4,096 bytes. The filesystem exposes 31,260,672 inodes, with 21,371 used and 31,239,301 free: there is no inode exhaustion. Marketing 512 GB is approximately 476.940 GiB before partition, encryption and filesystem overhead.

The 8.557-GiB mapped-device/capacity gap is filesystem overhead excluded from the capacity reported by statvfs; it is consistent with ext4 inode tables, journal and other internal structures, but their individual sizes could not be read. The 4.785-GiB free/available gap is reserved or otherwise privileged-only free space, not used dataset space and not a cleanup candidate. No filesystem settings were changed. Partition exterior space is 2,449,408 bytes and the LUKS mapping difference is 16,777,216 bytes. These values reconcile physical capacity; do not add filesystem overhead to reported used space twice.

### Coverage and measurement limits

Traversed 24,290 directory entries, 21,361 distinct listed inodes, 17,816 distinct regular-file inodes, 3,134 directories and 410 symbolic links. Allocated size is `st_blocks × 512`, deduplicated by `(st_dev, st_ino)`. No sparse regular files were observed. ext4 has no reflink sharing to estimate here; observed shared regular-file storage is through hardlinks. All regular-file link counts were accounted for by discovered aliases.

Root-owned `/mnt/codexa/lost+found` could not be enumerated; its directory itself is included. The unexplained used/listed gap is 2,105,344 bytes. It must not be advertised as deletable data: internal accounting, unreadable entries and live-filesystem timing can contribute. Detailed `tune2fs -l` access requires a password; passwordless elevation was unavailable. The scan did not change permissions or mount options.

Process inspection saw 654 PIDs, with 496 FD-directory denials and 87 memory-map denials. Among visible processes, **no deleted-open regular files on this filesystem** were found through FD or mapping inspection, and filesystem-filtered `lsof -nP -a +L1 /mnt/codexa` returned no entries. System-wide absence cannot be proved with these permissions. Deleted memfd/tmpfs objects from desktop applications are not SSD allocation. No live Codexa training, acquisition or pytest worker was visible; all six PIDs in the old pause record were absent. A future cleanup must recheck handles with sufficient visibility.

The scanner never followed directory symlinks, crossed filesystem boundaries, opened a database for writing, loaded CUDA, or deserialized untrusted legacy checkpoints. Current native-v1 baseline/continuation metadata was inspected separately with CPU-only `torch.load(weights_only=True, mmap=True)`. Most content reads used O_NOATIME where permitted; ordinary read-only fallback/diagnostic commands may update access timestamps under relatime. No file content, mtime, permissions or link topology was intentionally changed.

## Drive-wide directory breakdown

| Directory | Allocated bytes | GiB | Retention stance |
| --- | --- | --- | --- |
| `/mnt/codexa/pytorch/codexa-v3` | 157,061,644,288 | 146.275 | Protected current research |
| `/mnt/codexa/pytorch/workspace-outputs` | 138,959,945,728 | 129.417 | Retain unless individually reviewed |
| `/mnt/codexa/pytorch/saved` | 98,038,374,400 | 91.305 | Retain unless individually reviewed |
| `/mnt/codexa/numpy` | 40,027,058,176 | 37.278 | Retain unless individually reviewed |
| `/mnt/codexa/pytorch/relocated-extra` | 1,092,042,752 | 1.017 | Retain unless individually reviewed |
| `/mnt/codexa/pytorch/runs` | 4,096 | 0.000 | Retain unless individually reviewed |
| `/mnt/codexa/manifests` | 5,193,728 | 0.005 | Retain unless individually reviewed |
| `/mnt/codexa/.staging` | 131,072 | 0.000 | Retain unless individually reviewed |
| `/mnt/codexa/lost+found` | 16,384 | 0.000 | Unreadable contents; do not touch |

These sibling project trees do not share regular-file inodes with one another. Root files and parent-directory blocks form the small remaining allocation. Older migration reports establish that these are retained canonical assets moved onto this SSD; they are not automatically backup copies. In particular, a previous migration explicitly waived an independent backup. Prior cleanup authorization in those records does not authorize deletion now.

| Major subtree (nested; not additive) | Allocated bytes | GiB |
| --- | --- | --- |
| `/mnt/codexa/pytorch/workspace-outputs/chat-training` | 20,645,814,272 | 19.228 |
| `/mnt/codexa/pytorch/workspace-outputs/consolidation-smoke` | 9,662,464 | 0.009 |
| `/mnt/codexa/pytorch/workspace-outputs/investigation` | 16,908,288,000 | 15.747 |
| `/mnt/codexa/pytorch/workspace-outputs/preparation` | 31,918,977,024 | 29.727 |
| `/mnt/codexa/pytorch/workspace-outputs/preserved-models` | 7,477,284,864 | 6.964 |
| `/mnt/codexa/pytorch/workspace-outputs/qualification` | 28,182,065,152 | 26.247 |
| `/mnt/codexa/pytorch/workspace-outputs/training` | 33,817,849,856 | 31.495 |
| `/mnt/codexa/pytorch/saved/checkpoints` | 47,372,513,280 | 44.119 |
| `/mnt/codexa/pytorch/saved/data` | 36,122,980,352 | 33.642 |
| `/mnt/codexa/pytorch/saved/exports` | 14,542,852,096 | 13.544 |
| `/mnt/codexa/numpy/current/data` | 219,037,696 | 0.204 |
| `/mnt/codexa/numpy/current/data.before-20261013-cutover` | 2,281,472 | 0.002 |
| `/mnt/codexa/numpy/current/runs` | 39,516,119,040 | 36.802 |
| `/mnt/codexa/numpy/current/runs/diagnostics` | 3,039,703,040 | 2.831 |
| `/mnt/codexa/numpy/current/runs/pretraining` | 24,317,595,648 | 22.648 |
| `/mnt/codexa/numpy/current/runs/sft` | 12,158,812,160 | 11.324 |

### Largest 30 directories

Ranked by unique allocated blocks within each tree, excluding the mount root. Parent/child rows overlap and must not be summed.

| Rank | Directory | Allocated bytes | GiB |
| --- | --- | --- | --- |
| 1 | `/mnt/codexa/pytorch` | 395,152,015,360 | 368.014 |
| 2 | `/mnt/codexa/pytorch/codexa-v3` | 157,061,644,288 | 146.275 |
| 3 | `/mnt/codexa/pytorch/workspace-outputs` | 138,959,945,728 | 129.417 |
| 4 | `/mnt/codexa/pytorch/saved` | 98,038,374,400 | 91.305 |
| 5 | `/mnt/codexa/pytorch/codexa-v3/dataset-v2` | 87,398,457,344 | 81.396 |
| 6 | `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814` | 87,398,453,248 | 81.396 |
| 7 | `/mnt/codexa/pytorch/codexa-v3/pretraining` | 65,606,635,520 | 61.101 |
| 8 | `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/raw` | 48,070,737,920 | 44.769 |
| 9 | `/mnt/codexa/pytorch/saved/checkpoints` | 47,372,513,280 | 44.119 |
| 10 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201` | 40,564,441,088 | 37.779 |
| 11 | `/mnt/codexa/numpy` | 40,027,058,176 | 37.278 |
| 12 | `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/raw/_hub` | 39,884,705,792 | 37.146 |
| 13 | `/mnt/codexa/numpy/current` | 39,737,446,400 | 37.008 |
| 14 | `/mnt/codexa/numpy/current/runs` | 39,516,119,040 | 36.802 |
| 15 | `/mnt/codexa/pytorch/saved/data` | 36,122,980,352 | 33.642 |
| 16 | `/mnt/codexa/pytorch/workspace-outputs/training` | 33,817,849,856 | 31.495 |
| 17 | `/mnt/codexa/pytorch/workspace-outputs/preparation` | 31,918,977,024 | 29.727 |
| 18 | `/mnt/codexa/pytorch/workspace-outputs/preparation/codexa-900m-chat-10h-v1` | 31,918,972,928 | 29.727 |
| 19 | `/mnt/codexa/pytorch/saved/data/tokenized` | 28,942,864,384 | 26.955 |
| 20 | `/mnt/codexa/pytorch/saved/data/tokenized/base-v1` | 28,942,856,192 | 26.955 |
| 21 | `/mnt/codexa/pytorch/saved/data/tokenized/base-v1/mixed` | 28,942,835,712 | 26.955 |
| 22 | `/mnt/codexa/pytorch/workspace-outputs/qualification` | 28,182,065,152 | 26.247 |
| 23 | `/mnt/codexa/pytorch/workspace-outputs/qualification/codexa-934m-20261011-v1` | 28,182,061,056 | 26.247 |
| 24 | `/mnt/codexa/pytorch/workspace-outputs/qualification/codexa-934m-20261011-v1/gpu-recovery` | 28,182,056,960 | 26.247 |
| 25 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-real-01` | 25,042,190,336 | 23.322 |
| 26 | `/mnt/codexa/numpy/current/runs/pretraining` | 24,317,595,648 | 22.648 |
| 27 | `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/raw/_hub/fineweb-edu` | 23,536,750,592 | 21.920 |
| 28 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/curated` | 23,135,887,360 | 21.547 |
| 29 | `/mnt/codexa/pytorch/workspace-outputs/training/codexa-934m-foundation-5h-v1` | 22,545,657,856 | 20.997 |
| 30 | `/mnt/codexa/pytorch/workspace-outputs/training/codexa-934m-foundation-5h-v1/checkpoints` | 22,545,653,760 | 20.997 |

### Largest 50 individual files

One row per distinct regular-file inode, ranked by allocated bytes; equal-sized ties use canonical path order. Names in the same hardlink group are aliases, not extra files consuming the same bytes again. Logical length and allocated space are separately shown.

| Rank | Exact canonical path | Allocated bytes | Logical bytes | Links |
| --- | --- | --- | --- | --- |
| 1 | `/mnt/codexa/pytorch/saved/data/tokenized/base-v1/mixed/train.bin` | 25,639,702,528 | 25,639,694,468 | 1 |
| 2 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/curated/documents.sqlite` | 23,135,875,072 | 23,135,875,072 | 1 |
| 3 | `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/indexes/documents.sqlite` | 13,643,513,856 | 13,643,509,760 | 1 |
| 4 | `/mnt/codexa/pytorch/workspace-outputs/qualification/codexa-934m-20261011-v1/gpu-recovery/native/slot-0/checkpoint.pt` | 5,636,399,104 | 5,636,391,032 | 1 |
| 5 | `/mnt/codexa/pytorch/workspace-outputs/qualification/codexa-934m-20261011-v1/gpu-recovery/native/slot-1/checkpoint.pt` | 5,636,399,104 | 5,636,391,032 | 1 |
| 6 | `/mnt/codexa/pytorch/workspace-outputs/training/codexa-934m-foundation-5h-v1/checkpoints/slot-0/checkpoint.pt` | 5,636,399,104 | 5,636,394,168 | 1 |
| 7 | `/mnt/codexa/pytorch/workspace-outputs/training/codexa-934m-foundation-5h-v1/checkpoints/slot-1/checkpoint.pt` | 5,636,399,104 | 5,636,393,912 | 1 |
| 8 | `/mnt/codexa/pytorch/workspace-outputs/training/codexa-934m-foundation-5h-v1/checkpoints/slot-2/checkpoint.pt` | 5,636,399,104 | 5,636,394,104 | 1 |
| 9 | `/mnt/codexa/pytorch/workspace-outputs/training/codexa-934m-foundation-5h-v1/checkpoints/slot-3/checkpoint.pt` | 5,636,399,104 | 5,636,392,056 | 1 |
| 10 | `/mnt/codexa/pytorch/workspace-outputs/preparation/codexa-900m-chat-10h-v1/pilots/corrective-02/checkpoints/slot-3/checkpoint.pt` | 5,636,395,008 | 5,636,390,584 | 1 |
| 11 | `/mnt/codexa/pytorch/workspace-outputs/preparation/codexa-900m-chat-10h-v1/pilots/corrective-02/checkpoints/slot-4/checkpoint.pt` | 5,636,395,008 | 5,636,390,584 | 1 |
| 12 | `/mnt/codexa/pytorch/workspace-outputs/preparation/codexa-900m-chat-10h-v1/recovery-deterministic/recoverable/slot-1/checkpoint.pt` | 5,636,395,008 | 5,636,388,920 | 1 |
| 13 | `/mnt/codexa/pytorch/workspace-outputs/chat-training/codexa-900m-chat-2h-v1/checkpoints/slot-1/checkpoint.pt` | 5,636,075,520 | 5,636,070,993 | 1 |
| 14 | `/mnt/codexa/pytorch/workspace-outputs/chat-training/codexa-900m-chat-2h-v1/checkpoints/slot-2/checkpoint.pt` | 5,636,075,520 | 5,636,070,993 | 1 |
| 15 | `/mnt/codexa/pytorch/workspace-outputs/chat-training/codexa-900m-chat-2h-v1/checkpoints/slot-3/checkpoint.pt` | 5,636,075,520 | 5,636,070,993 | 1 |
| 16 | `/mnt/codexa/pytorch/workspace-outputs/investigation/chat-regression-20261010/pilots/repair-proportional-END3/checkpoints/slot-0/checkpoint.pt` | 5,636,075,520 | 5,636,070,993 | 1 |
| 17 | `/mnt/codexa/pytorch/workspace-outputs/investigation/chat-regression-20261010/pilots/repair-proportional-END3/checkpoints/slot-1/checkpoint.pt` | 5,636,075,520 | 5,636,070,993 | 1 |
| 18 | `/mnt/codexa/pytorch/workspace-outputs/investigation/chat-regression-20261010/pilots/repair-proportional-END3/checkpoints/slot-2/checkpoint.pt` | 5,636,075,520 | 5,636,070,993 | 1 |
| 19 | `/mnt/codexa/pytorch/workspace-outputs/preparation/codexa-900m-chat-10h-v1/pilots/corrective-02/checkpoints/slot-2/checkpoint.pt` | 5,636,075,520 | 5,636,070,737 | 1 |
| 20 | `/mnt/codexa/pytorch/workspace-outputs/preparation/codexa-900m-chat-10h-v1/recovery-deterministic/recoverable/slot-0/checkpoint.pt` | 5,636,075,520 | 5,636,069,073 | 1 |
| 21 | `/mnt/codexa/pytorch/workspace-outputs/training/codexa-934m-sft-feasibility-pilot-v1/checkpoints/slot-0/checkpoint.pt` | 5,636,075,520 | 5,636,069,073 | 1 |
| 22 | `/mnt/codexa/pytorch/workspace-outputs/training/codexa-934m-sft-feasibility-pilot-v1/checkpoints/slot-1/checkpoint.pt` | 5,636,075,520 | 5,636,069,073 | 1 |
| 23 | `/mnt/codexa/pytorch/saved/checkpoints/codexa-900m-base-v1/latest.pt` | 5,636,071,424 | 5,636,066,645 | 1 |
| 24 | `/mnt/codexa/pytorch/saved/checkpoints/codexa-900m-base-v1/previous.pt` | 5,636,071,424 | 5,636,066,645 | 1 |
| 25 | `/mnt/codexa/pytorch/saved/checkpoints/codexa-900m-chat-stage2-pilot-v1/best.pt` | 5,636,071,424 | 5,636,066,693 | 1 |
| 26 | `/mnt/codexa/pytorch/saved/checkpoints/codexa-900m-chat-stage2-pilot-v1/latest.pt` | 5,636,071,424 | 5,636,066,693 | 1 |
| 27 | `/mnt/codexa/pytorch/saved/checkpoints/codexa-900m-chat-stage2-pilot-v1/previous.pt` | 5,636,071,424 | 5,636,066,693 | 1 |
| 28 | `/mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458/raw/passage_counts.sqlite` | 3,972,857,856 | 3,972,857,856 | 1 |
| 29 | `/mnt/codexa/pytorch/workspace-outputs/chat-training/codexa-900m-chat-2h-v1/checkpoints/slot-0/checkpoint.pt` | 3,737,522,176 | 3,737,516,373 | 1 |
| 30 | `/mnt/codexa/pytorch/workspace-outputs/preparation/codexa-900m-chat-10h-v1/pilots/corrective-02/checkpoints/slot-0/checkpoint.pt` | 3,737,522,176 | 3,737,516,245 | 1 |
| 31 | `/mnt/codexa/pytorch/workspace-outputs/qualification/codexa-934m-20261011-v1/gpu-recovery/optimized/objects-store/objects/25/257829409647a979220ed7c1ee6b6b9d54af6b6ac34e34009de0eea23a30a168.pt` | 3,737,501,696 | 3,737,495,145 | 1 |
| 32 | `/mnt/codexa/pytorch/workspace-outputs/qualification/codexa-934m-20261011-v1/gpu-recovery/optimized/objects-store/objects/60/60d547f1a3a36d1a4bde9d7457aff9e5584d388ac859f88726b4f0dc3c383ad0.pt` | 3,737,501,696 | 3,737,495,145 | 1 |
| 33 | `/mnt/codexa/pytorch/workspace-outputs/qualification/codexa-934m-20261011-v1/gpu-recovery/optimized/objects-store/objects/87/879b99f3dac3ac4d929fefdfa3664efabfc9c130e260b070c0bb5fe7a80208ae.pt` | 3,737,501,696 | 3,737,495,145 | 1 |
| 34 | `/mnt/codexa/pytorch/saved/checkpoints/codexa-900m-sft-balanced-v1/latest.pt` | 3,737,497,600 | 3,737,489,848 | 1 |
| 35 | `/mnt/codexa/pytorch/saved/checkpoints/codexa-900m-sft-repair-v1/latest.pt` | 3,737,497,600 | 3,737,489,848 | 1 |
| 36 | `/mnt/codexa/pytorch/saved/checkpoints/codexa-900m-sft-sanity-v1/latest.pt` | 3,737,497,600 | 3,737,489,848 | 1 |
| 37 | `/mnt/codexa/pytorch/saved/checkpoints/codexa-900m-sft-v1/latest.pt` | 3,737,497,600 | 3,737,489,848 | 1 |
| 38 | `/mnt/codexa/pytorch/saved/checkpoints/codexa-900m-sft-v2/latest.pt` | 3,737,497,600 | 3,737,489,848 | 1 |
| 39 | `/mnt/codexa/pytorch/saved/exports/codexa-900m-chat-stage2-pilot-v1/model.pt` | 3,737,489,408 | 3,737,482,887 | 1 |
| 40 | `/mnt/codexa/pytorch/workspace-outputs/preserved-models/20261010-controlled-01/exports/balanced/model.pt` | 3,737,489,408 | 3,737,482,887 | 1 |
| 41 | `/mnt/codexa/pytorch/workspace-outputs/preserved-models/20261010-controlled-01/exports/repair/model.pt` | 3,737,489,408 | 3,737,482,887 | 1 |
| 42 | `/mnt/codexa/pytorch/saved/data/processed/fineweb-edu-1b-v1/train.jsonl` | 3,733,000,192 | 3,732,993,482 | 1 |
| 43 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/raw/fineweb-edu-expanded.jsonl` | 3,608,059,904 | 3,608,051,945 | 1 |
| 44 | `/mnt/codexa/pytorch/saved/data/tokenized/base-v1/mixed/train_index.json` | 3,303,124,992 | 3,303,120,860 | 1 |
| 45 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-real-01/curated/documents.sqlite` | 3,113,873,408 | 3,113,873,408 | 1 |
| 46 | `/mnt/codexa/numpy/current/runs/pretraining/pretrain_250m_clean_fp32_500step/checkpoint_step_000250.npz` | 3,039,690,752 | 3,039,683,896 | 1 |
| 47 | `/mnt/codexa/numpy/current/runs/pretraining/pretrain_250m_clean_fp32_500step/checkpoint_step_000500.npz` | 3,039,690,752 | 3,039,683,900 | 1 |
| 48 | `/mnt/codexa/numpy/current/runs/pretraining/pretrain_250m_clean_fp32_500step/target_final.npz` | 3,039,690,752 | 3,039,683,900 | 1 |
| 49 | `/mnt/codexa/numpy/current/runs/pretraining/pretrain_250m_clean_full/checkpoint_step_004548.npz` | 3,039,690,752 | 3,039,683,860 | 1 |
| 50 | `/mnt/codexa/numpy/current/runs/pretraining/pretrain_250m_clean_full/checkpoint_step_009096.npz` | 3,039,690,752 | 3,039,683,860 | 1 |

## Codexa v3: what occupies the space

| Subtree | Allocated bytes | GiB | Purpose |
| --- | --- | --- | --- |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2` | 87,398,457,344 | 81.396 | Phase-2 sources, indexes, working tokens, snapshots, bundles and reviews |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201` | 40,564,441,088 | 37.779 | 500M corpus, full native checkpoints, training logs and checks |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-real-01` | 25,042,190,336 | 23.322 | 100M pilot corpus, five full pilot checkpoints, terminal/recovery validation and test scratch |
| `/mnt/codexa/pytorch/codexa-v3/diagnostics` | 4,056,547,328 | 3.778 | Forward-only diagnostic outputs, scripts and auxiliary passage index |

### Preparation root components

| Component under `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814` | Inclusive bytes | GiB | Exclusive bytes if whole tree unlinked |
| --- | --- | --- | --- |
| benchmarks | 9,609,216 | 0.009 | 9,609,216 |
| cache | 275,386,368 | 0.256 | 275,386,368 |
| clean | 4,096 | 0.000 | 4,096 |
| continuation | 5,971,243,008 | 5.561 | 5,971,243,008 |
| indexes | 13,720,915,968 | 12.779 | 13,720,915,968 |
| logs | 868,352 | 0.001 | 868,352 |
| raw | 48,070,737,920 | 44.769 | 45,842,976,768 |
| reports | 18,141,184 | 0.017 | 18,141,184 |
| reviews | 9,875,456 | 0.009 | 9,875,456 |
| snapshots | 14,667,132,928 | 13.660 | 14,667,132,928 |
| tmp | 2,460,499,968 | 2.292 | 232,738,816 |
| tokens | 4,421,505,024 | 4.118 | 4,421,505,024 |

These inclusive component rows are **not additive**: raw TinyStories and its temporary inspection name share one 2,227,761,152-byte allocation. The whole preparation root is 87,398,453,248 allocated bytes. Directory-inclusive accounting differs slightly from older inventories that counted files only.

### Downloaded sources

| Source; flat raw namespace | Payload names | Allocated bytes | GiB | Status |
| --- | --- | --- | --- | --- |
| cosmopedia-educational | 2 | 450,478,080 | 0.420 | Rejected-source evidence retained |
| cosmopedia-stories | 1 | 276,840,448 | 0.258 | Rejected-source evidence retained |
| finemath-4plus | 16 | 4,597,252,096 | 4.282 | Retain pinned source payloads |
| fineweb | 1 | 645,926,912 | 0.602 | Unadmitted partial; P1 |
| fineweb-edu | 13 | 26,365,448,192 | 24.555 | Retain pinned source payloads |
| openwebmath | 1 | 243,052,544 | 0.226 | Retain pinned source payloads |
| tinystories-gpt4 | 2 | 2,250,264,576 | 2.096 | Retain pinned source payloads |
| wikipedia-en | 41 | 11,631,058,944 | 10.832 | Retain pinned source payloads |

Flat source payloads exclude their receipts and hidden transport state. The raw tree also contains the two old interrupted HF transfers S1/S2 and transport metadata. The 411 registry entries describe candidate source files; they are not 411 completed downloads. No new data was acquired by this audit. FineWeb is distinct from FineWeb-Edu. Cosmopedia payloads remain evidence for the documented source rejection; being excluded from training does not make them expendable.

The 500M run additionally retains a 21.547-GiB curated database/index tree, 3.775 GiB raw inputs and 1.655 GiB token arrays. These are historical data dependencies for training exposure, diagnostics and phase-2 exclusion checks, not merely duplicate download caches. Its curated/documents.sqlite is the second-largest file on the whole drive. Phase 2 has another 12.779-GiB index tree with new compressed source records, duplicate/protection structures and provenance. Derived uint16 arrays are a separate representation, not byte-identical copies of those databases.

### Snapshot rebuild costs and shared shards

| Snapshot name under `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots` | Inclusive allocated bytes | GiB | New bytes introduced in shown order | Exclusive bytes now |
| --- | --- | --- | --- | --- |
| stage-0001 | 232,747,008 | 0.217 | 232,747,008 | 232,747,008 |
| reviewed-2b-20261105 | 4,731,760,640 | 4.407 | 4,731,760,640 | 4,184,834,048 |
| remediated-v1-20261105 | 1,404,928 | 0.001 | 1,404,928 | 1,404,928 |
| remediated-v1b-20261105 | 4,671,102,976 | 4.350 | 4,260,909,056 | 4,260,909,056 |
| remediated-v2-20261105 | 4,687,994,880 | 4.366 | 4,141,068,288 | 1,321,578,496 |
| remediated-v3-20261105 | 4,665,655,296 | 4.345 | 1,299,238,912 | 1,299,238,912 |

All snapshots combined consume **14,667,128,832 bytes (13.660 GiB)**. The first two snapshots together consumed 4.624 GiB. Later attempts added 9.036 GiB of actual allocation: v1b introduced 3.968 GiB, v2 3.857 GiB and v3 1.210 GiB.

Version 1 stopped before token payload publication; its roughly 1.34-MiB policy/quarantine record is failed-operation evidence, not a hidden multi-gigabyte dataset. Versions 1b and 2 are near-full extra token/mapping allocations. Version 3 shares most unchanged payloads with version 2, adding changed math shards and a new full document mapping. The current v3 tree has 4,665,655,296 inclusive bytes but only 1,299,238,912 bytes exclusive to it. Removing one shared name would not free its shared shard.

The rebuilder explicitly hardlinks unchanged bin/index pairs and copies retained token bytes into new files for changed shards. It does not copy raw datasets into each snapshot. The finalizer separately materializes tokenized snapshots from working tokens. Each successful remediation also writes its own full documents.jsonl, manifests, quarantine and policy evidence. Those document mappings contribute hundreds of MiB per version. No retained solution is silently rewritten.

### Reference and dependency trace

| Manifest | SHA256 | Referenced payload/index paths present | Remediation parent hash |
| --- | --- | --- | --- |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/manifest.json` | 7d1a9e38589988113d795019e4aa3e081a032551fd79025a38c1180926f2eab3 | 68 | 5cc849cc50680c86a50c40fe10aa6f56d0b0cfc77e1dcab5feb7ddba8c5dfddf |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v2-20261105/manifest.json` | 156bd2d07b95ddd45a136cb884d58aed178efcfeab90829f0ede7efb7a6f1fe3 | 68 | 5cc849cc50680c86a50c40fe10aa6f56d0b0cfc77e1dcab5feb7ddba8c5dfddf |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v3-20261105/manifest.json` | 199183a06b130912d87712e4d857a4d706649909de59d4c55b665a89e83d2c85 | 68 | 156bd2d07b95ddd45a136cb884d58aed178efcfeab90829f0ede7efb7a6f1fe3 |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/reviewed-2b-20261105/manifest.json` | 5cc849cc50680c86a50c40fe10aa6f56d0b0cfc77e1dcab5feb7ddba8c5dfddf | 68 | None |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/stage-0001/manifest.json` | 3ee1cbcdc7cfef0f3cc3c7772f568ad07eb21c05ac607ecc5b7279ca58ea4233 | 8 | None |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/tokens/manifest.json` | fb5dda98d2002fdf468da94b9428d09b7e2198b752ce5d9aab4396c024c2ff28 | 3434 | None |

All 3,714 referenced payload/index path occurrences exist. No manifest was published for the incomplete v1 directory. The working manifest, original reviewed snapshot, source index and pinned tokenizer form the reproduction base. V2 names the original snapshot as parent and explicitly binds v1b as reused screening input in policy.json. V3 names v2 as its parent. This is why an old-looking candidate cannot be classified as obsolete from timestamps alone.

The three frozen review protocols bind, respectively, the original reviewed-2b, v2 and v3 manifest hashes. Baseline held-out measurements also bind stage-0001 and reviewed-2b. Preserve these parent/evaluation snapshots and their manifests. P2 is only a conditional proposal for v1b bin/index payloads; deleting them currently breaks historical verify/reuse-screen commands and requires an explicit reproduction decision. No dependency is bypassed by relabelling data as old.

### Checkpoints, optimizer state and training runs

CPU-only native checkpoint inspection found the original 500M checkpoint at 15,259 updates and 500,006,912 targets. Its unique model tensor storage is **994,749,952 bytes**; optimizer tensors occupy **1,989,500,584 bytes across 170 parameter states**. The roughly 2.78-GiB checkpoint is mostly necessary model/Adam state, not disposable logging. Removing optimizer contents would alter recovery behavior and is outside this audit.

Each continuation initial.pt stores the same model and preserved optimizer tensors with different phase metadata/fingerprints. Both are at phase step zero and 500,006,912 cumulative tokens. The older bundle binds original reviewed-2b; the exact-50M bundle binds remediated-v3 and specifies 50,000,000 additional targets. They consume 5.561 GiB together and are **not interchangeable or byte-identical**. They remain MUST KEEP. No continuation training was started.

| Artifact/tree | Inclusive allocated bytes | GiB | Notes |
| --- | --- | --- | --- |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/final/checkpoint.pt` | 2,985,615,360 | 2.781 | Immutable 500M reference; three hardlink names share one inode |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/final/best-validation-step-15000.pt` | 2,985,615,360 | 2.781 | Best-validation state; two hardlink names share one inode |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/training/step-015250-1793777133791449722.pt` | 2,985,615,360 | 2.781 | Previous recovery state; P7 needs retention approval |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-real-01/pilot` | 14,932,742,144 | 13.907 | Five full checkpoint files; final/latest duplicated; P3–P6 are conditional |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-real-01/reference-pty` | 2,987,569,152 | 2.782 | Distinct 20-update terminal/exact-recovery validation state; retain evidence |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/continuation/healthcheck-preserve` | 2,985,619,456 | 2.781 | Historical untrained continuation; preserve native fingerprints |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/continuation/remediated-v3-exact50m` | 2,985,619,456 | 2.781 | Current exact-50M untrained continuation; quality gate failed |

Baseline SHA256 reverified: `54fca007f272fb3784c759d18db44912091227aedc233b62c9731514b1835c73`. Frozen tokenizer SHA256 reverified: `60a5e06447542b9462d458577f4a3056551c135e5aa1b8859d39624a3eee8771`. The tokenizer has deliberate separate copies for fitting/reproduction proof; do not retire them simply because their hashes match.

### Caches, temporary artifacts, hidden data and logs

The preparation cache root is only 0.256 GiB; it is not the cause of tens or hundreds of GiB of use. Its apparent child totals also share blocks: plot-env and the uv archive contain hardlinked installed packages. Deleting the uv cache alone would free only 6,307,840 bytes, not its 144,142,336 inclusive bytes. Preserve the plotting environment and Xet acquisition logs until their reproduction roles are resolved. Empty CUDA/Inductor/pip roots have negligible allocation.

The selected remediated-v3 snapshot contains 2,076,912,557 training, 10,432,298 validation and 10,335,976 test positions. Its complete fresh quality review failed; selected/current here means retained preparation candidate, not an approved or running training dataset.

| Cache subtree; overlapping children noted above | Inclusive bytes | Exclusive bytes |
| --- | --- | --- |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/cache/cuda` | 4,096 | 4,096 |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/cache/hf` | 16,572,416 | 16,572,416 |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/cache/inductor` | 4,096 | 4,096 |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/cache/matplotlib` | 131,072 | 131,072 |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/cache/mypy` | 40,947,712 | 40,947,712 |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/cache/pip` | 4,096 | 4,096 |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/cache/plot-env` | 139,259,904 | 1,425,408 |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/cache/python` | 72,040,448 | 72,040,448 |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/cache/ruff` | 90,112 | 90,112 |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/cache/uv` | 4,096 | 4,096 |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/cache/xdg` | 144,158,720 | 6,324,224 |
| `/mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458/cache/cuda` | 4,096 | 4,096 |
| `/mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458/cache/matplotlib` | 131,072 | 131,072 |
| `/mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458/cache/mypy` | 40,812,544 | 40,812,544 |
| `/mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458/cache/pycache` | 13,025,280 | 13,025,280 |
| `/mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458/cache/pytest` | 28,672 | 28,672 |
| `/mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458/cache/ruff` | 77,824 | 77,824 |
| `/mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458/cache/xdg` | 16,384 | 16,384 |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/cache/cuda` | 4,096 | 4,096 |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/cache/mypy` | 40,824,832 | 40,824,832 |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/cache/pycache` | 61,345,792 | 61,345,792 |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/cache/pytest` | 36,864 | 36,864 |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/cache/ruff` | 77,824 | 77,824 |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/cache/torchinductor` | 4,096 | 4,096 |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/cache/xdg` | 634,880 | 634,880 |

Hidden `raw/_hub/*/.cache/huggingface/download` directories contain partial payloads and receipt/lock metadata; the word cache does not make the whole tree safe to delete. Likewise `raw/_hub` has 37.146 GiB inclusive but only 1.500 GiB exclusive, mostly the two incomplete transfers. Most completed payloads are hardlinked to flat raw names. Deleting those hub aliases alone frees no payload blocks while the flat names remain.

Do not wipe an entire tmp tree: it contains source-acquisition helpers, review_io scripts, sampling specifications and preserved registry candidates. The 2.075-GiB tiny-prefix-inspection.txt is a hardlink to the verified raw TinyStories train source; removing that alias alone recovers **zero** payload bytes. `tmp/test-v2-current` also contains test harness material whose provenance is less explicit, so it is not in the safe total. Only the exact pytest-generated roots S3–S6 are proposed as low risk. Zero-length lock files and small SQLite -wal/-shm files are not meaningful savings; database sidecars are excluded from all cleanup candidates.

The main 500M logs tree occupies 1.854 GiB, almost entirely the 1,990,602,752-byte ANSI recording P8. Structured training/validation metrics, event streams, supervisor/install logs and reports must remain. The diagnostic raw tree includes a 3.700-GiB auxiliary SQLite repetition index. It is bound in artifact_sha256.json and the diagnostic report; retain it as required evidence. Re-creating that index would be a separate operation, not a safe cache cleanup approved by this audit.

## Classification and cleanup candidates

**SAFE TO DELETE** below means low-risk content identified by this audit, not authorization to delete it now. Process visibility is incomplete; repeat full handle/dependency checks immediately before any later approved cleanup. **POSSIBLY SAFE** means unresolved retention/dependency decisions prevent approval. **MUST KEEP** protects unique/current recovery, scientific evidence and required reproduction inputs. Old age, a cache-like name, equal file length, or a failed quality gate alone never proves dispensability.

### SAFE TO DELETE — 3.099 GiB, pending your approval

| ID | Exact path | Allocated bytes | Recoverable bytes | Risk |
| --- | --- | --- | --- | --- |
| S1 | `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/raw/_hub/wikipedia-en/.cache/huggingface/download/20231101.en/GU92ZH035Ac_wOOPpZVTiqaFZdI=.16ff05ab829cb9c983db14b7f77a677d279f8120571e67b2793735851b71ae89.6a1036cc.incomplete` | 134,189,056 | 134,189,056 | Low |
| S2 | `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/raw/_hub/fineweb-edu/.cache/huggingface/download/sample/10BT/BsnyBITQ1GHtauZLSNvyrADtxIo=.4b7d1f697d4afff7f5f65cb7aa4d83acc6db716da4d46ea23399d0c50608c6e3.9ef218b2.incomplete` | 1,475,461,120 | 1,475,461,120 | Low |
| S3 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-real-01/tmp/pytest-of-k9-vortex` | 809,631,744 | 809,631,744 | Low |
| S4 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/tmp/pytest` | 204,152,832 | 204,152,832 | Low |
| S5 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/checks/postrun-20261104-073346/pytest-tmp` | 204,152,832 | 204,152,832 | Low |
| S6 | `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/tmp/pytest-of-k9-vortex` | 230,146,048 | 230,146,048 | Low |
| S7 | `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/cache/python` | 72,040,448 | 72,040,448 | Low |
| S8 | `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/cache/mypy` | 40,947,712 | 40,947,712 | Low |
| S9 | `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/cache/ruff` | 90,112 | 90,112 | Low |
| S10 | `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/cache/matplotlib` | 131,072 | 131,072 | Low |
| S11 | `/mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458/cache/pycache` | 13,025,280 | 13,025,280 | Low |
| S12 | `/mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458/cache/mypy` | 40,812,544 | 40,812,544 | Low |
| S13 | `/mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458/cache/ruff` | 77,824 | 77,824 | Low |
| S14 | `/mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458/cache/matplotlib` | 131,072 | 131,072 | Low |
| S15 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/cache/pycache` | 61,345,792 | 61,345,792 | Low |
| S16 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/cache/mypy` | 40,824,832 | 40,824,832 | Low |
| S17 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/cache/ruff` | 77,824 | 77,824 | Low |
| S18 | `/mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458/cache/pytest` | 28,672 | 28,672 | Low |
| S19 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/cache/pytest` | 36,864 | 36,864 | Low |

**S1.** Interrupted Hugging Face transfer; complete source is now retained. No exact artifact dependency or visible open handle; pinned complete counterpart verified: /mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/raw/wikipedia-en--20231101.en--train-00038-of-00041.parquet; SHA256 16ff05ab829cb9c983db14b7f77a677d279f8120571e67b2793735851b71ae89 What is lost: Only transfer progress; retain complete source, receipt and pause/acquisition logs. Regeneration: Retained complete source needs no download. A future transfer can recreate partial data.

**S2.** Interrupted Hugging Face transfer; complete source is now retained. No exact artifact dependency or visible open handle; pinned complete counterpart verified: /mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/raw/fineweb-edu--sample--10BT--007_00000.parquet; SHA256 4b7d1f697d4afff7f5f65cb7aa4d83acc6db716da4d46ea23399d0c50608c6e3 What is lost: Only transfer progress; retain complete source, receipt and pause/acquisition logs. Regeneration: Retained complete source needs no download. A future transfer can recreate partial data.

**S3–S6:** Finished pytest-generated fixtures and tiny synthetic checkpoints. Known test roots; example run.json specifies 133,440 parameters and CPU. Verification reports/logs remain outside candidate. No current test worker visible. What is lost: Scratch fixtures and synthetic states; no original model weights or scored benchmark results. Regeneration: Tests regenerate functionally equivalent fixtures; run IDs/timestamps differ. Do not run tests during audit. The four exact roots are independent allocations; repeated fixture contents were not added again as duplicate-file savings. Sampled run metadata differentiates the 133,440-parameter test model from the real 248,687,488-parameter model.

**S7–S19:** Each exact path is a generated tool cache named in its row: Python bytecode, mypy analysis, Ruff, Matplotlib font discovery, or pytest bookkeeping. Source/configuration/environment and final check outputs remain outside these paths. What is lost is warm-start performance or transient test bookkeeping; each installed tool regenerates its own cache. No download, tokenizer retraining, dataset rebuilding or model training is needed. HF logs/receipts, uv plotting packages and source snapshots are deliberately excluded.

### POSSIBLY SAFE — 25.502 GiB additional, not approved

| ID | Exact path or specifically listed set | Allocated bytes | Recoverable bytes | Risk |
| --- | --- | --- | --- | --- |
| P1 | `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/raw/fineweb--sample--10BT--002_00000.parquet.partial` | 645,926,912 | 645,926,912 | Low/medium; confirm acquisition recovery is unnecessary. |
| P2 | `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105` — only the 68 payload paths listed in Appendix A | 4,201,234,432 | 3,791,040,512 | High; dependency/reproducibility approval required. |
| P3 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-real-01/pilot/step-3031.pt` | 2,985,615,360 | 2,985,615,360 | Medium; duplicate bytes do not prove path dispensable. |
| P4 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-real-01/pilot/smoke-20.pt` | 2,985,615,360 | 2,985,615,360 | Medium/high; explicit historical-retention decision. |
| P5 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-real-01/pilot/step-1000.pt` | 2,985,615,360 | 2,985,615,360 | Medium/high; explicit historical-retention decision. |
| P6 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-real-01/pilot/step-2000.pt` | 2,985,615,360 | 2,985,615,360 | Medium/high; explicit historical-retention decision. |
| P7 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/training/step-015250-1793777133791449722.pt` | 2,985,615,360 | 2,985,615,360 | Medium/high; retain final and best-validation states. |
| P8 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/logs/terminal.ansi` | 1,990,602,752 | 1,990,602,752 | Medium; evidence-retention approval. |
| P9 | `/mnt/codexa/pytorch/saved/checkpoints/codexa-900m-chat-stage2-pilot-v1/previous.pt` | 5,636,071,424 | 5,636,071,424 | Medium/high; other-generation owner/dependency approval. |
| P10 | `/mnt/codexa/pytorch/saved/exports/codexa-100m-native-v1-lmstudio-f16.gguf` | 277,467,136 | 277,467,136 | Medium; other-generation export approval. |
| P11 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/packages/cache` | 112,992,256 | 112,992,256 | Low/medium; permissions and retention approval. |

**P1:** Paused acquisition of unadmitted general FineWeb, distinct from FineWeb-Edu. **References/status:** Dataset report explicitly says paused/preserved; mixture excludes it. Old paused PIDs are gone; future acquisition could resume it. **Would lose:** 645,922,816 bytes of transfer progress. **Regeneration:** Restart pinned download, subject to availability and separate authorization.

**P2:** Superseded candidate token/index payloads only; preserve all manifests, policy, document listing, quarantine and audit evidence. **References/status:** Version-2 policy explicitly records this manifest as reused screening input. Removing payloads breaks historical verify/reuse-screen commands; reconstruction not tested here. **Would lose:** Immediate byte verification/reuse of the 68 historical payload files listed below. **Regeneration:** Potentially reconstructible from frozen policy, original snapshot, tokenizer and index; exact matching unverified.

**P3:** Separate byte-identical copy of pilot/latest.pt. **References/status:** Both paths appear in pilot report; latest also referenced by pilot-compatibility check. Keep latest, SHA256 e3164d9ddf6d855833e48e920f4cbc06c89560c0f8c838d4e814cd3497a18706. **Would lose:** Historical filename, not unique model bytes. **Regeneration:** Bytes retained in latest; resolve removed-filename consumers first.

**P4:** Intermediate checkpoint from completed approximately 100M pilot. **References/status:** Historical pilot outputs; not 500M baseline or continuation parent. Historical state-retention requirements unresolved. **Would lose:** Exact model, optimizer, RNG and sampler state at that step. **Regeneration:** Requires historical training replay; exact recreation not guaranteed and prohibited in this audit.

**P5:** Intermediate checkpoint from completed approximately 100M pilot. **References/status:** Historical pilot outputs; not 500M baseline or continuation parent. Historical state-retention requirements unresolved. **Would lose:** Exact model, optimizer, RNG and sampler state at that step. **Regeneration:** Requires historical training replay; exact recreation not guaranteed and prohibited in this audit.

**P6:** Intermediate checkpoint from completed approximately 100M pilot. **References/status:** Historical pilot outputs; not 500M baseline or continuation parent. Historical state-retention requirements unresolved. **Would lose:** Exact model, optimizer, RNG and sampler state at that step. **Regeneration:** Requires historical training replay; exact recreation not guaranteed and prohibited in this audit.

**P7:** Previous recovery point before final step 15,259. **References/status:** Sidecar identifies this state; final and best-validation states are separate and preserved. Comparison/recovery obligations need checking. **Would lose:** Exact 15,250-update model, optimizer, RNG and sampler state. **Regeneration:** Training replay required; exact recreation not guaranteed.

**P8:** Full ANSI dashboard repaint recording from six-hour run. **References/status:** Historical terminal evidence; structured metrics/events and final reports retained separately. **Would lose:** Exact terminal playback, not structured loss/token history. **Regeneration:** Exact original stream cannot be regenerated.

**P9:** Separate byte-identical copy of best.pt in another generation. **References/status:** SHA256 f59884878d4a98439b593751add5feeae8a34fe4d3c904af531cb21c37576454; keep best.pt. Older rollback filename consumers unresolved. **Would lose:** previous.pt recovery alias. **Regeneration:** Bytes retained in best.pt; resolve filename dependencies first.

**P10:** Separate byte-identical export under alternate name. **References/status:** SHA256 347e6744fda5fd925e75ca5e96b8ba81999a34894a065731e6d20278344065b3; keep codexa-100m-base-v1-lmstudio-llama-f16.gguf. Runtime path dependencies unresolved. **Would lose:** Alternate export filename, not unique model bytes. **Regeneration:** Bytes retained in canonical twin; check consumers first.

**P11:** DNF repository metadata staged for monitoring-terminal installation. **References/status:** Installation report records package versions; offline provenance/recovery requirements unresolved. Some files root-owned. **Would lose:** Offline historical repository metadata. **Regeneration:** Functional cache refreshable; exact old metadata and remote availability not guaranteed.

P2 would not free all of its inclusive 4.201-GB payload allocation: some shards have aliases in other snapshots. Its exact selector is the 68 bin/index paths, not the whole snapshot directory. Keep every policy, manifest, quarantine record, documents.jsonl and audit file. Even then, historical verification/reuse operations would fail until payloads were restored; the current classification is high-risk and conditional. No such retirement has been approved.

P3/P9/P10 are content duplicates but retain distinct names used or potentially used by historical consumers. This audit proposes retaining the named canonical twin; it does not authorize deleting aliases, replacing files with hardlinks, editing manifests or changing recovery configurations. P4–P7 are unique historical states and are conditional retention tradeoffs, not redundant data. P8 cannot be reproduced exactly.

### MUST KEEP and excluded from savings

| Exact path/tree | Why retained |
| --- | --- |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/final/checkpoint.pt` | Original 500M model, optimizer, RNG, sampler and native resume metadata. Preserve every required baseline path. |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/final/best-validation-step-15000.pt` | Distinct best-validation state; retain verified model selection/recovery evidence. |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/tokenizer` | Frozen tokenizer and training provenance. Identical tokenizer copies are fitting/reproduction evidence. |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v3-20261105` | Current selected candidate and held-out data; quality failure does not make it disposable. |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/continuation` | Both verified untrained native continuation bundles/configurations and optimizer state. |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/reviews` | All frozen blind packets, ratings, decisions, source mappings, sampling/integrity/rendering records; failed and unscored historical evidence included. |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/reports` | Independent verification, baseline integrity, source snapshots, review evidence and acquisition/registry history. |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/reviewed-2b-20261105` | Original review/bundle/evaluation input and remediation ancestor. |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v2-20261105` | Frozen failed review input and current v3 byte-verification parent. |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/stage-0001` | Bound baseline held-out evaluation input; duplicate first shard is not approval to remove it. |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1-20261105` | Incomplete attempt's unique policy/quarantine evidence; no large token payload to reclaim. |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/tokens` | Working shards bound by working-manifest identity and finalization reproduction. |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/indexes` | Source records, historical/new/evaluation duplicate indexes, progress and protection state. Dataset SQLite sidecars are not cleanup candidates. |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/raw` | Retain verified source payloads/receipts except individually reviewed partials. Rejected Cosmopedia files are source-quality evidence. |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/curated` | Historical corpus and index needed for original exposure/provenance, diagnostic replay and historical exclusion. |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/raw` | Original training inputs/provenance; overlap with pilot data is not proof whole files are duplicates. |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/tokens` | Original training/evaluation token streams for exact resumption and matched held-out comparison. |
| `/mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458` | Diagnostic scripts, source snapshots, results, figures, samples and checksums; especially raw/passage_counts.sqlite. Only listed generated caches are candidates. |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/training` | Retain final/best aliases, metrics.jsonl, events.jsonl, summary and run metadata; only P7 is conditional. |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/checks` | Retain scientific/checkpoint/validation reports and outputs; S6 is generated test scratch only. |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-real-01/pilot/latest.pt` | Keep one complete final pilot state even if P3 filename retirement is later approved. |
| `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-real-01/reference-pty` | Actual terminal/exact-recovery validation artifacts; not equivalent to synthetic pytest scratch. |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/tmp` | Retain review/acquisition/protection helpers, specifications and registry evidence; never blanket-clean tmp. Only S3 root is selected here. |
| `/mnt/codexa/manifests` | Historical migration, retained-originals and recovery proofs; do not replay their old cleanup instructions. |
| `/mnt/codexa/.staging` | Tiny historical migration staging/evidence, not a material space-recovery target. |
| `/mnt/codexa/pytorch/workspace-outputs` | Unique other-generation runs, recovery objects, reports and exports retained by default. |
| `/mnt/codexa/pytorch/saved` | Canonical other-generation datasets, models and exports; only specifically identified P9/P10 are conditional. |
| `/mnt/codexa/numpy` | Canonical NumPy datasets, models, adapters and migration links; no blanket cleanup proposal. |
| `/mnt/codexa/pytorch/relocated-extra` | Other-generation artifacts and benchmark evidence; duplicates need their own ownership/dependency decision. |
| `/mnt/codexa/lost+found` | Unreadable contents; filesystem recovery area, not an audit-approved deletion target. |

Broad protected trees above have only the specifically enumerated S/P exceptions; the rest stays protected. Hypothetically removing whole older project trees could free far more bytes, but would discard unique model/data assets. Their full sizes are **not included** in recoverable totals. No lack of a visible current process establishes that an older model is unwanted.

## Duplicate and hardlink findings

There are **2,914 hardlinked regular-file inodes**. Summing apparent bytes over all regular-file names would overstate storage by **53,912,893,401 bytes (50.210 GiB)** before directory/block rounding. No hardlink group had undiscovered aliases according to its link count. Symlinks were counted as links only; their targets were not charged again.

Key examples: final/checkpoint.pt, training/latest.pt and training/step-015259-1793777155957645512.pt share one 2,985,615,360-byte allocation. The final best-validation alias and training/step-015000-1793776779254211637.pt share another. TinyStories raw/tmp names and most flat/_hub download pairs also share inodes. Removing any one of these payload aliases while keeping the others returns zero payload blocks. Path consumers may still break.

The duplicate-content scan first grouped distinct inodes by length. For files at least 1 MiB it screened 299 candidates in 62 size groups with 64-KiB start/middle/end fingerprints, then fully SHA256-hashed 167 surviving candidate files (19,018,686,820 bytes). It also fully hashed 12,185 size-matched smaller files (245,607,291 bytes). Prefix hashes were never treated as proof of equality. Full-content matching found **2,899 duplicate groups**, including 32 groups at least 1 MiB. Empty files are ignored as content-deduplication candidates.

Keeping one distinct inode per matched group gives a purely mathematical redundant-block ceiling of **9,950,031,872 bytes (9.267 GiB)**. This includes protected tokenizer/evaluation copies, scientific fixtures and path-dependent checkpoints; it is **not an approved savings total** and is not added to candidate totals. Already-shared hardlinks were excluded before duplicate-content hashing.

| Confirmed SHA256 | Distinct inode paths | Maximum per-copy allocation; not approval |
| --- | --- | --- |
| f59884878d4a98439b593751add5feeae8a34fe4d3c904af531cb21c37576454 | `/mnt/codexa/pytorch/saved/checkpoints/codexa-900m-chat-stage2-pilot-v1/best.pt`<br>`/mnt/codexa/pytorch/saved/checkpoints/codexa-900m-chat-stage2-pilot-v1/previous.pt` | 5,636,071,424 |
| e3164d9ddf6d855833e48e920f4cbc06c89560c0f8c838d4e814cd3497a18706 | `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-real-01/pilot/step-3031.pt`<br>`/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-real-01/pilot/latest.pt` | 2,985,615,360 |
| 347e6744fda5fd925e75ca5e96b8ba81999a34894a065731e6d20278344065b3 | `/mnt/codexa/pytorch/saved/exports/codexa-100m-base-v1-lmstudio-llama-f16.gguf`<br>`/mnt/codexa/pytorch/saved/exports/codexa-100m-native-v1-lmstudio-f16.gguf` | 277,467,136 |
| fd28d73380d79cd8b458428321d3280d7749eb55f1b906d6d6f6f03801cac5d2 | `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/reviewed-2b-20261105/train-00000.bin`<br>`/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/stage-0001/train-00000.bin` | 134,225,920 |

The first snapshot training shard in stage-0001 and reviewed-2b is byte-identical but allocated separately; both are protected evaluation/reproduction artifacts. Identically named or similarly sized checkpoints across other 900M/934M runs generally did not pass full-content equality screening. Matching dimensions, model tensors alone or file lengths do not establish duplicate optimizer/recovery state. The audit did not attempt tensor-level deduplication, checkpoint conversion or model redesign.

## Savings and recommended approval order

| Conditional cumulative scenario | Cumulative bytes recovered | Projected available bytes | Projected available GiB |
| --- | --- | --- | --- |
| Current available | 0 | 62,579,019,776 | 58.281 |
| After all SAFE candidates | 3,327,303,680 | 65,906,323,456 | 61.380 |
| After SAFE plus P1/P11 (paused transfer and installer metadata) | 4,086,222,848 | 66,665,242,624 | 62.087 |
| Then P3/P9/P10 (duplicate aliases, after dependency clearance) | 12,985,376,768 | 75,564,396,544 | 70.375 |
| Then P4–P7 (unique historical checkpoint retention waived) | 24,927,838,208 | 87,506,857,984 | 81.497 |
| Then P8 (terminal recording retention waived) | 26,918,440,960 | 89,497,460,736 | 83.351 |
| Then P2 (historical snapshot payload dependency resolved) | 30,709,481,472 | 93,288,501,248 | 86.882 |

Totals: safely recoverable **3,327,303,680 bytes (3.099 GiB)**; additional potentially recoverable **27,382,177,792 bytes (25.502 GiB)**; selected combined potential **30,709,481,472 bytes (28.600 GiB)**. Proposed entries and their inode memberships were checked for overlap: each category's union equals the sum of its recoverable rows. The scenario table assumes every named conditional candidate is eventually approved; it is not a claim all should be deleted.

Approve in the displayed order, beginning with stale transfer remnants, regenerated tool caches and completed synthetic test scratch. Resolve legacy filename consumers before duplicate-alias retirement. Treat unique historical checkpoints, raw terminal evidence and v1b payloads as separate high-impact retention decisions. Do not delete original/reviewed/current datasets, continuation bundles or source databases to make a nominal savings target. Do not lower ext4 reserves, change the 40-GiB training free-space floor, or alter the 84-GiB preparation ceiling as part of cleanup.

Recoverable figures use allocated inode blocks, including directories only where the whole specified directory is retired. Parent-directory shrinkage, journal effects, delayed reclamation, concurrent writes and newly opened handles can change measured free-space gains. These forecasts preserve the currently privileged-only reserve. No claimed recovery was measured because no files were deleted.

## Verification and reproducible read-only commands

A second full inventory under the project's Python 3.13.14 independently reproduced the first stdlib scanner's totals and coverage (initial scanner Python 3.15.0). GNU du independently returned exactly **435,184,422,912 allocated bytes**, with the same lost+found permission warning. Final volume capacity/used/available counters matched the initial measurements. All inventoried inode identities, allocated blocks, sizes, link counts, mtimes and ctimes were rechecked; none changed, and no new volume path appeared. Access timestamps were not used for this assertion.

The original checkpoint and tokenizer were fully SHA256-hashed again at the end and matched their initial expected hashes. The current remediated-v3 manifest, all 68 referenced bin/index payloads (4,221,485,062 logical bytes) and documents.jsonl were hashed and matched their manifest entries. Its document mapping SHA256 is `2a770b0b5c20fa58083b512867035c1bd8ca44c25f6270d9088530a57ea7409c`. These storage/integrity checks do not reverse the failed corpus-quality gate. No inference, backward pass, optimizer update, training test, database rewrite or training supervisor was run.

Metadata dependency inspection covered 926 small JSON/YAML/TOML/Markdown/script/hash files outside cache/tmp trees, capped at 2 MiB each, plus targeted source, reports and native checkpoint metadata. Required manifests were structurally traced, and source-receipt/hash relationships were checked. This is not exhaustive analysis of dynamically constructed paths, every historical event stream, inaccessible processes or every other generation's external runtime configuration. Consequently unresolved legacy consumers are POSSIBLY SAFE rather than safe. Historical documents containing old cleanup instructions were treated only as evidence.

Re-run from the repository working directory. These commands print measurements and do not clean or alter data:

```sh
findmnt -T /mnt/codexa -o TARGET,SOURCE,FSTYPE,UUID,OPTIONS
findmnt -R /mnt/codexa -o TARGET,SOURCE,FSTYPE
lsblk -b -o NAME,TYPE,SIZE,FSTYPE,UUID,MOUNTPOINTS /dev/sda
df -B1 /mnt/codexa
df -i /mnt/codexa
stat -f -c 'block=%S total=%b free=%f available=%a inodes=%c free_inodes=%d' /mnt/codexa
du -x -B1 --max-depth=3 /mnt/codexa
lsof -nP -a +L1 /mnt/codexa
```

Do not sum separate du invocations across linked trees; their independent totals can charge the same inode twice. The following standalone inventory reconstructs whole-volume inode allocation without following symlinks, also reports permission gaps, and prints the rankings. It keeps everything in RAM:

```sh
.venv/bin/python -B - <<'PY'
import sys,json,os,stat,collections,hashlib,time,subprocess,traceback
from pathlib import Path
G=1024**3
root="/mnt/codexa"
expected="279352d0-0011-4733-8348-6815a4308196"
assert subprocess.check_output(["findmnt","-T",root,"-n","-o","UUID"],text=True).strip()==expected
started=time.time(); device=os.stat(root).st_dev
entries={}; inodes={}; directories={}; errors=[]; symlinks=[]
stack=[root]
while stack:
 d=stack.pop()
 try:
  s=os.lstat(d)
  entries[d]={"dev":s.st_dev,"ino":s.st_ino,"blocks":s.st_blocks*512,"size":s.st_size,"links":s.st_nlink,"mode":s.st_mode,"mtime":s.st_mtime_ns,"ctime":s.st_ctime_ns}
  directories[d]=set()
  with os.scandir(d) as it: children=list(it)
 except OSError as e:
  errors.append({"path":d,"error":str(e)});continue
 for child in children:
  p=child.path
  try:s=child.stat(follow_symlinks=False)
  except OSError as e:errors.append({"path":p,"error":str(e)});continue
  if s.st_dev!=device:
   errors.append({"path":p,"error":"different filesystem; not traversed"});continue
  row={"dev":s.st_dev,"ino":s.st_ino,"blocks":s.st_blocks*512,"size":s.st_size,"links":s.st_nlink,"mode":s.st_mode,"mtime":s.st_mtime_ns,"ctime":s.st_ctime_ns}
  entries[p]=row
  if stat.S_ISLNK(s.st_mode):
   row["target"]=os.readlink(p);symlinks.append(p)
  elif stat.S_ISDIR(s.st_mode):stack.append(p)
for p,s in entries.items():
 key=(s["dev"],s["ino"])
 item=inodes.setdefault(key,{"blocks":s["blocks"],"size":s["size"],"links":s["links"],"mode":s["mode"],"paths":[]})
 item["paths"].append(p)
 parent=p if stat.S_ISDIR(s["mode"]) else os.path.dirname(p)
 while parent.startswith(root):
  directories.setdefault(parent,set()).add(key)
  if parent==root:break
  parent=os.path.dirname(parent)
def allocated(keys):return sum(inodes[k]["blocks"] for k in keys)
def inside(path,prefix):return path==prefix or path.startswith(prefix.rstrip("/")+"/")
def reclaim(paths):
 paths=list(paths);result=[]
 for k,item in inodes.items():
  if all(any(inside(p,q) for q in paths) for p in item["paths"]):
   if stat.S_ISREG(item["mode"]) and item["links"]>len(item["paths"]):continue
   result.append(k)
 return allocated(result)
def tree(path):
 keys=directories.get(path,set())
 return {"path":path,"allocated_bytes":allocated(keys),"exclusive_bytes":reclaim([path]),"regular_files":sum(len(inodes[k]["paths"]) for k in keys if stat.S_ISREG(inodes[k]["mode"])),"apparent_unique_bytes":sum(inodes[k]["size"] for k in keys if stat.S_ISREG(inodes[k]["mode"]))}
def emit(x):print(json.dumps(x,ensure_ascii=False),flush=True)
v=os.statvfs(root)
filesystem={"frsize":v.f_frsize,"blocks":v.f_blocks,"bfree":v.f_bfree,"bavail":v.f_bavail,"capacity":v.f_blocks*v.f_frsize,"used":(v.f_blocks-v.f_bfree)*v.f_frsize,"available":v.f_bavail*v.f_frsize,"reserved_free":(v.f_bfree-v.f_bavail)*v.f_frsize,"inodes":v.f_files,"free_inodes":v.f_ffree}
files=sorted([i for i in inodes.values() if stat.S_ISREG(i["mode"])],key=lambda x:x["blocks"],reverse=True)
emit({"ready":True,"started_epoch":started,"files":len(files),"directory_entries":len(entries),"directories":len(directories),"symlinks":len(symlinks),"errors":errors,"filesystem":filesystem,"scanned_allocated":allocated(inodes),"hardlinked_inodes":sum(i["links"]>1 for i in files),"top_major":[tree(p) for p in [root+"/pytorch/codexa-v3",root+"/pytorch/workspace-outputs",root+"/pytorch/saved",root+"/numpy",root+"/pytorch/relocated-extra"]]})
emit({"top30_directories": sorted([(p,allocated(k)) for p,k in directories.items() if p != root],key=lambda x:(-x[1],x[0]))[:30]})
emit({"top50_files": [{"canonical_path":sorted(i["paths"])[0],"allocated_bytes":i["blocks"],"logical_bytes":i["size"],"aliases":sorted(i["paths"])[1:]} for i in sorted(files,key=lambda i:(-i["blocks"],sorted(i["paths"])[0]))[:50]]})
PY
```

Native checkpoint storage can be inspected without GPU execution using the existing project interpreter and CPU-only mmap loading. No tensor or optimizer contents were stripped; the model/optimizer storage figures are observations of existing native files, not a conversion proposal. For process-wide deleted-open absence or an ext4 journal/inode-table breakdown, a later sufficiently privileged read-only inspection is needed. No password prompt, permission change, remount, filesystem repair or reserve adjustment was attempted during implementation.

## Appendix A — exact P2 payload set

Base candidate is high-risk historical snapshot payload retirement only. All nonlisted metadata/evidence remains protected. Each row is an exact pathname; allocated bytes may be shared with names outside this set. The combined recoverable payload allocation is **3,791,040,512 bytes**, not the sum of inclusive rows.

| Exact payload path | Allocated bytes | Other aliases outside this set |
| --- | --- | --- |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/test-00000.bin` | 20,578,304 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/test-00000.bin.idx` | 131,072 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00000.bin` | 132,546,560 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00000.bin.idx` | 499,712 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00001.bin` | 133,361,664 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00001.bin.idx` | 630,784 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00002.bin` | 132,665,344 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00002.bin.idx` | 540,672 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00003.bin` | 133,681,152 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00003.bin.idx` | 1,761,280 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00004.bin` | 134,221,824 | Yes |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00004.bin.idx` | 2,510,848 | Yes |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00005.bin` | 134,221,824 | Yes |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00005.bin.idx` | 2,510,848 | Yes |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00006.bin` | 134,217,728 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00006.bin.idx` | 2,510,848 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00007.bin` | 134,221,824 | Yes |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00007.bin.idx` | 2,506,752 | Yes |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00008.bin` | 133,427,200 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00008.bin.idx` | 970,752 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00009.bin` | 132,833,280 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00009.bin.idx` | 368,640 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00010.bin` | 132,304,896 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00010.bin.idx` | 507,904 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00011.bin` | 132,571,136 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00011.bin.idx` | 512,000 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00012.bin` | 132,337,664 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00012.bin.idx` | 503,808 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00013.bin` | 132,665,344 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00013.bin.idx` | 499,712 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00014.bin` | 133,603,328 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00014.bin.idx` | 593,920 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00015.bin` | 98,349,056 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00015.bin.idx` | 139,264 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00016.bin` | 126,922,752 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00016.bin.idx` | 552,960 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00017.bin` | 133,447,680 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00017.bin.idx` | 598,016 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00018.bin` | 133,394,432 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00018.bin.idx` | 647,168 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00019.bin` | 133,361,664 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00019.bin.idx` | 626,688 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00020.bin` | 133,603,328 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00020.bin.idx` | 647,168 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00021.bin` | 133,431,296 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00021.bin.idx` | 684,032 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00022.bin` | 131,235,840 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00022.bin.idx` | 237,568 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00023.bin` | 132,743,168 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00023.bin.idx` | 389,120 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00024.bin` | 132,542,464 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00024.bin.idx` | 598,016 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00025.bin` | 130,584,576 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00025.bin.idx` | 491,520 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00026.bin` | 132,739,072 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00026.bin.idx` | 516,096 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00027.bin` | 132,567,040 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00027.bin.idx` | 516,096 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00028.bin` | 132,743,168 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00028.bin.idx` | 524,288 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00029.bin` | 132,583,424 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00029.bin.idx` | 520,192 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00030.bin` | 132,063,232 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00030.bin.idx` | 581,632 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00031.bin` | 52,568,064 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/train-00031.bin.idx` | 196,608 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/val-00000.bin` | 20,733,952 | No |
| `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/val-00000.bin.idx` | 135,168 | No |

## Final audit status

Inspection finished 2026-11-05T14:59:41.201160+00:00. **No deletion, move, cleanup, source-data edit, checkpoint modification, dataset rebuild or training was performed.** Only the requested report is created in the repository. Existing unrelated repository changes are preserved. Low-risk and conditional candidates remain present for your review; there is no cleanup authorization in this report.
