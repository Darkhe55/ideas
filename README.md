# 深入创想（Ideas）

[简体中文](README.md) | [English](README.en.md)

记录一些想法、论证过程与最终方案。以文档为主，必要的关键代码随文档内嵌。

## 目录结构

```
.
├── README.md          中文说明（本文件）
├── README.en.md       English README
└── docs/
    ├── 简略说明版/    草稿：不超过 2 页
    ├── 扩展中途版/    扩充中：超过 2 页、不足 15 页
    ├── 最终版/        成稿：超过 15 页
    └── en/            英文文档（与中文同层级）
        ├── brief/
        ├── extended/
        └── final/
```

三个阶段文件夹是**同一份文档的三个阶段**，不是三份文档；中英文各保留一份同名文档，互为镜像。

## 收纳规则

- **每种语言只留一份**：同一想法在同一语言下任一时刻只有一份文档，三档不并行、不重复收纳；
- 草稿（不超过 2 页）放在简略说明版；扩充超过 2 页后移到扩展中途版；继续续写超过 15 页后移到最终版；
- 迁移用 `git mv` 移动文件，文件名不变；早期阶段由 git 历史保留，不在工作区留副本；
- 文档头部标注当前「阶段」，迁移时同步更新；
- 英文版本放在 `docs/en/` 下，阶段对应关系：brief ↔ 简略说明版、extended ↔ 扩展中途版、final ↔ 最终版。

## 当前文档

| 想法 | 阶段 | 中文 | English |
| --- | --- | --- | --- |
| 快速预测框架 | 最终版 | [快速预测框架](docs/最终版/快速预测框架.md) | [Fast Prediction Framework](docs/en/final/fast-prediction-framework.md) |
| 我如何知道我不知道什么 | 扩展中途版 | [我如何知道我不知道什么](docs/扩展中途版/我如何知道我不知道什么.md) | [How Do I Know What I Don't Know](docs/en/extended/how-do-i-know-what-i-dont-know.md) |
| MLP 的结构升阶 | 扩展中途版 | [MLP 的结构升阶](docs/扩展中途版/MLP的结构升阶.md) | [Higher-Order Structure of MLP](docs/en/extended/higher-order-structure-of-mlp.md) |
| Transformer 的隐式信息表示 | 扩展中途版 | [Transformer 的隐式信息表示](docs/扩展中途版/Transformer的隐式信息表示.md) | [Implicit Information Representation in Transformers](docs/en/extended/implicit-information-representation-in-transformers.md) |

## 写作约定

- 关键代码用 PyTorch 伪代码，追求「能看懂、能照着写」，不追求可直接运行。
- 用相对路径互链；中文为主，术语首次出现时附英文；中英文档内容保持一致。
- 新的想法从简略说明版开始写，随内容增长逐级迁移。

## 安全

本仓库为**公开仓库**，不提交任何密钥、令牌、账号密码、个人隐私信息与私有数据。`.gitignore` 已排除常见敏感文件；提交前请自查：

```bash
git status
git diff --cached
```

## 同步

```bash
git remote add origin ssh://git@ssh.github.com:443/Darkhe55/ideas.git
git push -u origin main
```
