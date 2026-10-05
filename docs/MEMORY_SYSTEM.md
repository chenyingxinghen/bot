# 记忆系统

## 架构与数据隔离

记忆实现位于 `core/memory/`，由 `core/conversation/service.py` 编排，配置集中在 `config/memory.py`。

三种模式使用独立数据库：

- `data/memory_tavern.db`
- `data/memory_clone.db`
- `data/memory_writer.db`

路径由 `MEMORY_DB=data/memory_{mode}.db` 控制。若旧配置未写 `{mode}`，程序会自动插入模式后缀。

每个模式内继续按以下作用域隔离：

- bot 账号：`self_id`
- QQ 私聊对象：`peer_id`
- tavern 当前角色：`char:<slug>`
- writer 当前作品：`work:<slug>`
- clone：无第三维 namespace

短期上下文、runtime messages、互动时间、在线抽取游标和长期记忆均按模式/namespace 处理，避免切模式、角色或作品后串味。`person` 类型表示关于用户本人的稳定事实，可在同一模式的角色/作品之间共享；`self/relationship/episode` 按 namespace 隔离。

## 在线抽取与召回

运行中的用户消息和机器人回复写入当前模式数据库的 `messages`。满足消息数、对方消息数和语义字符阈值后，在线抽取任务提交到全局 `InferenceQueue`，避免绕过回复/生图并发限制。

抽取结果分为：

- `self`：角色/作者自身设定
- `person`：关于用户的事实和偏好
- `relationship`：双方互动关系
- `episode`：事件或剧情

保存时生成 embedding，并使用高阈值语义去重；embedding 服务异常时仍保存正文，之后可用回填工具补齐。

记忆注入分为两层：

1. **基础记忆胶囊**：每轮固定注入当前业务作用域内高置信、未过期且状态为 `active` 的 `person/self/relationship`。三种类型分别设配额并共享字符预算；`episode` 和 `disputed` 不进入常驻层，避免旧事件或未决冲突持续影响当前回复。
2. **当前话题补充**：按需使用 embedding 向量排名、中文词面和关键词排名、时间意图以及 Reciprocal Rank Fusion，从 `self/person/relationship/episode` 中选择当前话题候选。embedding 不可用时自动退回词面和时间召回，不阻断回复。

两层都先按 `kind + subject_id` 成对验证业务作用域，避免历史脏数据仅因 subject 相同而误入。低信息消息（如“嗯”“好”“哈哈”）不发起当前话题检索，但仍会获得基础记忆胶囊；“继续”“然后呢”等指代型短句会带最近两条上下文构造查询。时间类问题会主动加入当前作用域最近的 `episode`，不要求正文关键词命中。

合并时基础胶囊优先，当前话题补充逐条去重后进入剩余预算。clone、tavern、writer 分别使用独立的胶囊类型配额、胶囊字符预算、话题召回数量、话题字符预算、总字符预算和向量阈值。

在线抽取同样按业务场景解释证据：clone 不把机器人回复当作用户事实；tavern 的虚构关系与剧情只进入当前角色 namespace；writer 的人物、事件和伏笔只进入当前作品 namespace，小说人物不得误记为用户。

## QQ 命令

### 查看

- `/记忆`：查看当前角色、作品或当前会话的有效记忆。
- `/记忆 模式`：查看当前 QQ 用户在当前模式的全部有效记忆；不会展示其他用户。
- `/记忆 全部`：模式层兼容写法，与 `/记忆 模式` 相同。

### 清理

- `/清记忆`：清当前角色/作品；保留关于用户的 `person` 记忆和其他 namespace。
- `/清记忆 模式`：清当前 QQ 用户在当前模式的全部记忆；不影响其他用户。
- `/清记忆 全部`：模式层兼容写法。

所有聊天命令都保留原始对话消息。清理时抽取游标会推进到当前作用域最后一条历史消息，因此旧消息不会自动重新生成刚清掉的记忆。

`MemoryStore.clear_all()` / `list_all()` 是后台运维接口，不暴露给普通 QQ 用户。

## 离线工具

按目标模式显式指定数据库：

```powershell
python tools/memory/prepare_conversations.py --db data/memory_clone.db --peer <QQ号>
python tools/memory/extract_memories.py --db data/memory_clone.db --self-id <机器人QQ> --partner <对方QQ>
python tools/memory/backfill_embeddings.py --db data/memory_clone.db
python tools/memory/resolve_conflicts.py --db data/memory_clone.db
```

`prepare_conversations.py` 只整理工作数据库，不删除原始导出文件；`backfill_embeddings.py` 可重复运行，只处理缺失或正文变化的向量。

## 清理与冲突语义

- `duplicate`：重复记忆，不参与普通查看和召回。
- `historical`：被新状态取代的历史事实。
- `disputed`：未决冲突，仍可召回并明确提示。
- `valid_to` 已过期的记录不参与普通查看和召回。

原始消息和长期记忆是两类数据：保留原始消息不等于允许自动重新抽取已清理历史，抽取游标负责维持这一边界。
