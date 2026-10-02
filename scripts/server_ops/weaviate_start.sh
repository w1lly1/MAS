#!/bin/bash
# 修复 Weaviate 的 "attempting to join" 死循环（8080 永不监听）。
#
# 现象：节点不停地想 join 自己的地址。原因是 weaviate-data 里的集群元数据
# 记录的是**当年那个容器名**，而 AutoDL 每次开机容器名可能不同；
# CLUSTER_HOSTNAME 给成"当前 hostname"时对不上，于是死不罢休。
#
# 做法：先从数据目录里把**记录的容器名**挖出来，用它当 CLUSTER_HOSTNAME，
#       再配 RAFT_BOOTSTRAP_EXPECT=1 按单节点启动。
set -u
echo "--- 当前 hostname ---"
hostname
echo
echo "--- 数据目录里记录过的容器名 ---"
grep -rao 'autodl-container[a-zA-Z0-9-]*' /root/autodl-tmp/weaviate-data 2>/dev/null \
  | sed 's/.*://' | sort | uniq -c | sort -rn | head -10
echo
echo "--- 停掉当前死循环进程 ---"
pkill -f 'weaviate --host' && echo "killed" || echo "(nothing to kill)"
sleep 3
pgrep -af 'weaviate --host' | head -2 || echo "confirmed stopped"
echo
RECORDED=$(grep -rao 'autodl-container[a-zA-Z0-9-]*' /root/autodl-tmp/weaviate-data 2>/dev/null \
  | sed 's/.*://' | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
NAME="${RECORDED:-$(hostname)}"
echo "--- 用 CLUSTER_HOSTNAME=$NAME 重启 ---"
cd /root/autodl-tmp/MAS || exit 1
nohup env AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  PERSISTENCE_DATA_PATH=/root/autodl-tmp/weaviate-data \
  CLUSTER_HOSTNAME="$NAME" \
  RAFT_BOOTSTRAP_EXPECT=1 \
  /root/autodl-tmp/MAS/weaviate --host 0.0.0.0 --port 8080 --scheme http \
  > /root/autodl-tmp/weaviate.log 2>&1 &
echo "pid=$!"
for i in $(seq 1 40); do
  # 注意：/v1/.well-known/ready **返回 200 但 body 是空的**，用 `grep '{}'` 判会永远等满 80 秒
  # （这是个已记录在案的假阴性）——所以这里判 HTTP 状态码。
  CODE=$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://localhost:8080/v1/.well-known/ready 2>/dev/null)
  if [ "$CODE" = "200" ]; then
    echo "READY after $((i*2))s (http $CODE)"; break
  fi
  sleep 2
done
echo "ready http: $(curl -s -o /dev/null -w '%{http_code}' -m 5 http://localhost:8080/v1/.well-known/ready)"
echo "listener: $(ss -lnt 2>/dev/null | grep -c 8080) on 8080"
echo
echo "--- 最后 6 条日志 ---"
tail -6 /root/autodl-tmp/weaviate.log | cut -c1-260
