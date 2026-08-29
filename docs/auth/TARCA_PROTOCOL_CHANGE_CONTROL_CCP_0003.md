# TARCA 协议变更控制记录 CCP-0003

> 状态：`APPROVED_AND_APPLIED`
> 授权日期：2026-08-29
> 稳定协议身份：`TARCA-E2E-STAGE-PROTOCOL-2.0`
> 变更性质：Stage1B 单一 v2 身份与冻结覆盖规则

## 1. 用户决定

Stage1B 不再把确认运行称为活动 `r1`、`r2` 或其他递增修订。项目只保留一个活动科学身份：
`v2`。此前名称中的确认轮次仅是原始实验来源标识，不是活动版本号，也不得出现在当前冻结
目录、活动指针或后续研究合同中。

本决定不删除真实历史：首次完整 pilot 的失败事实、确认运行的原始配置名和原始
`qualification_id` 仍保留在历史文档或不可篡改收据中。它们只用于追溯，不形成多个活动 v2。

## 2. 活动冻结结构

唯一活动冻结位于：

```text
artifacts/stage1b/active.json
artifacts/stage1b/frozen/v2/qualification_receipt.json
artifacts/stage1b/frozen/v2/manifest.json
artifacts/stage1b/frozen/v2/manifest.sha256
```

`active.json` 只记录 `series: v2` 和活动 manifest 的 SHA-256。manifest 不含 revision 字段；
科学身份由世界、模型、配置、来源、执行证据、资格收据及其内容哈希共同确定。

## 3. 正常冻结与授权覆盖

- 正常情况下，已存在的 `frozen/v2` 禁止修改或覆盖；
- 失败或不完整运行不得冻结，也不得登记成新的历史活动版本；
- 用户可以明确授权修改或覆盖，但授权必须同时记录授权人、原因和当前活动 manifest 的精确
  SHA-256；
- 覆盖前必须重新验证当前冻结、旧 manifest 哈希和新收据，发布采用临时目录与原子替换；
- 任一校验或发布步骤失败时，原活动 v2 必须保持可验证；
- 授权覆盖后，活动身份仍叫 `v2`，不创建递增编号。

## 4. 本次冻结决定

用户已审阅并授权把最终服务器确认结果设为 Stage1B v2。冻结门禁为 `PASS`，活动 manifest
SHA-256 为：

```text
d1b4d09260bcc41b3b94a020474ee0b5e9f9dd5f0f498bb96510228141f44b25
```

资格收据 SHA-256 为：

```text
d17a6523dbe4e1e82d6ed36c2a27982f0b54d64146d0cdcbcaa61b6407d2aef0
```

活动目标是 `lorenz96_twoscale_v2 + ITransformerReference` 在 h1–6 上的概率预测能力。
h7–12 只作次要诊断，h13–24 是已知能力边界；它们不被扩大解释为“所有时距都优于 VAR”。

## 5. 协议边界

本变更只完成 Stage1B 资格与冻结，不运行 E01 或 E02，不修改 Stage0/Stage1A 的科学身份，
也不放宽后续模型解释性实验的预注册、负对照、zero-refit 或因果声明边界。

若本文件与 CCP-0002 的活动版本命名或冻结目录说法冲突，以本文件为准；CCP-0002 的 pilot
观察、确认范围、种子、门禁和防窥视要求仍作为历史实验依据保留。
