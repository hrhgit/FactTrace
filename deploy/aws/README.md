# FactTrace 服务部署模板

此目录只保留当前主线的服务模板。主入口与本地一致：

```bash
python -m text_memory serve --env-file /home/ubuntu/.config/text-memory/service.env --database /home/ubuntu/.local/share/text-memory/memory.sqlite --host 127.0.0.1 --port 8766
```

安装依赖用本目录的 `requirements.txt`。环境文件填写 `MEMORY_*`、向量配置和 `MEMORY_SERVICE_TOKEN`，由部署者提供可访问的模型渠道；主线不另指定模型。模板路径可按实际机器调整。

`text-memory.service` 使用同一命令；`nginx.conf` 仅转发 `/health`、`/add`、`/search`。启动前检查目录权限、环境变量和接口密钥，正常配置失败会报错，不启动另一套算法。HTTP 认证与协议见 [主线说明](../../docs/collections.md)。

本次只整理本地模板，没有上传代码、重启远端服务、申请评测或提交版本。此前部署状态与检查记录已随旧说明归档；它们不证明这份源码已部署。
