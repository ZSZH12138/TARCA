# TARCA 协议变更控制记录 CCP-0001

> 状态：`APPROVED`
> 授权日期：2026-08-20
> 协议文档修订：`v2.0` → `v2.0.1`
> 稳定协议身份：`TARCA-E2E-STAGE-PROTOCOL-2.0`
> 变更性质：兼容性勘误与缺失授权边界补全

## 1. 触发原因

Stage 0→Stage 1A 交接复核发现两个协议问题：

1. 协议把 `Sha256Hash` 写成带 `sha256:` 前缀，但代码、测试、交接快照及全部冻结 Stage 0 artifact 一致使用 64 位小写十六进制；
2. 协议引用 `SealedAccessGrant`，但没有定义其字段，也没有让 Stage 1A 的物理读取函数接收 grant。

## 2. 已批准决策

### 2.1 SHA-256 wire format

以当前冻结 artifact 已经使用的表示为 canonical wire format：

```text
64 lowercase hexadecimal characters
```

不带算法前缀。协议文档修订为 v2.0.1，但稳定协议身份保持 2.0，因为本次变更纠正文字与既有 wire representation 的不一致，不改变实际序列化内容。

### 2.2 Sealed access

新增严格、冻结的 `SealedAccessGrant`，绑定：

- grant ID；
- 精确 dataset name/version；
- scope；
- 允许读取的 partition；
- `SEALED_ACCESS_AUTHORIZATION` 类型的授权 ArtifactRef；
- 签发和过期 UTC 时间。

`build_windows()` 与 `hash_dataset()` 增加可选 grant；sealed 读取必须在物理 I/O 前验证 dataset、scope、partition 和时间，任一不匹配都 fail closed。grant 只授权读取，不能授权 test-time fit、访问未来标签或修改 scientific identity。

effective sealed 状态由 registry 与请求 scope 共同决定；registry 已标记 `sealed=True` 时，调用方不得通过传入 `AccessScope(sealed=False)` 将其降级为 unsealed。

## 3. Artifact 与 Gate 影响

本次变更：

- 不重新生成 Stage 0 artifact；
- 不修改任何现有 JSON 文件字节或 content hash；
- 不替换 `ResearchContractManifest`；
- 不重签 Gate 0；
- 不替换 `Stage0CompletionReceipt`；
- 不改变预注册、新颖性声明、术语、假设、依赖锁或第三方来源边界。

因此不触发科学身份变化或新颖性复核。若未来选择把 wire format 改成带前缀的形式，则属于不兼容迁移，必须另立 CCP、升级稳定协议身份并受控重开 Stage 0。

## 4. 实现边界

本 CCP 只补充公共数据访问契约与纯校验函数。Dataset registry、loader、`WindowBatch`、Arrow/Parquet、SCM、模型、干预、OT 和 DRO 仍由对应 Stage 实现，不因本次修订被视为已完成。

## 5. 验证要求

- raw 64 位 hash 可通过，带前缀形式被拒绝；
- grant 严格、冻结、拒绝额外字段；
- dataset logical key、partition 集合、授权类型和有效期被校验；
- sealed 访问对缺失、过期和不匹配 grant fail closed；
- Stage 0 原有门禁、测试、静态检查和 lock check 继续通过。
