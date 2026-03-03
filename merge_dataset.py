import pyarrow as pa
import pyarrow.parquet as pq
import random

def load_and_clean(path):
    table = pq.read_table(path)
    # 如果存在 __index_level_0__，就删掉
    if "__index_level_0__" in table.column_names:
        table = table.drop(["__index_level_0__"])
    return table

def shuffle_table(table):
    num_rows = table.num_rows
    indices = list(range(num_rows))
    random.shuffle(indices)
    return table.take(pa.array(indices))

t1 = load_and_clean("data/aime2024/train.parquet")
t2 = load_and_clean("data/gsm8k/train.parquet")
t1_shuffled = shuffle_table(t1)
t2_sampled = t2.slice(0, 1920)
t2_shuffled = shuffle_table(t2_sampled)

merged = pa.concat_tables([t1, t2_sampled])
merged = shuffle_table(merged)
pq.write_table(merged, "data/aime_gsm8k_merged/train.parquet")
print(f"✅ 合并成功！总行数: {merged.num_rows}")