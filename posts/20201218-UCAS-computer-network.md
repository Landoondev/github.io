# 国科大计算机网络 20-21 秋季

中国科学院大学研究生的计算机网络课程（2020 年 秋季学期）。

## 授课老师

有 4 为老师授课，一位来自中科院计算机网络信息中心，其余三位来自中科院计算所。

## 课程资源

教学课件大约有 20 份。

## 考核方式

- 大作业 30%：我做的内容是实现一个 HTTPS 服务器
- 前沿学术讨论 20% ：阅读论文，我选择的是 [A Computational Approach to Packet Classification](https://landodo.github.io/posts/2020-12-03-Paper-for-computer-network)
- 平时课堂10%
- 考试 40%

## 课程内容

比本科丰富得多，新的东西非常多，有一种听不懂的感觉。

课程用几周的时间讲了本科阶段的计算机网络基础，然后就进入到专题部分，4 个老师讲其各自的研究内容，以专题的方式进行授课。

## 第一讲 计算机网络概述

第一讲主要介绍了计算机网络的体系结构、性能、安全以及挑战和机遇。并且还讲了中科院在网络方面的相关工作。

互联网不仅仅改变生活与社会，还推动信息技术本身发展。

物理传输：通信

- 基础对象：bit
- 调制、编码、信道、同步
- 香农定理：$C=Blog_2(1 + \frac{S}{N})$，C：传输速率；B：信道带宽；S/N 为信噪比。
- 调幅、调频、调相

数据传输：包交换

- 基础对象：数据帧
- 存储转发

包交换 VS 链路交换。

数据传输：介质访问控制 MAC（medium access control）

网络互连：异构网络，不同物理寻址机制。

路由与路由查找：路由协议（控制平面）、路由查找转发（数据平面）

可靠性与传输控制：如何在不可靠路径上尽力而为的实现可靠数据传输？无差错、不丢失、不重复、顺序。如何实现拥塞控制？接收窗口（接收端流控）；拥塞窗口（发送端流控）、拥塞判断：丢包、延迟、多指标。

DNS：递归查询

层次结构与实现模块化

![](./20201218/2.jpeg)

### 功能放置

```c++
// 客户端
SocketFD = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
connect(SocketFD, (struct sockaddr *)&sa, sizeof sa);
send() and recv()
write() and read()
close();

// 服务器
listen(SocketFD, QUEUE);
ConnectFD = accept(SocketFD, NULL, NULL);
send() and recv()
write() and read();
close();
```

报文封装：HTTP Header、TCP Header、IP Header、Ethernet Header。

延迟构成：$T=T_{tras}+T_{proc}+T_{prop}+T_{queue}$

- $T_{tras}=2P/R$：传输延迟
- $T_{proc}$：查找处理
- $T_{prop}$：信号传播延迟
- $T_{queue}$：排队延迟

吞吐量

![](./20201218/3.jpeg)

### BGP

1989，3 张餐巾纸中的边界网关协议（BGP）[图片来源: YAKOV REKHTER/WASHINGTON POST](https://www.stuff.co.nz/technology/digital-living/69048160/the-three-napkins-protocol-quick-fix-for-early-internet-problem-left-web-open-to-attack)。

![](./20201218/4.jpg)

网络的安全性和可靠性：路由安全、DDoS、安全防护困难。

### 中科院与中国互联网发展

- 1975 年，计算所开始关注网络研究
- 1981 年，计算所成立了网络研究室(十室)是国内最早开始从
  事互联网研究的实验室
- 1983 年，中科院与德国弗朗霍夫信息与生物技术研究所合作 研制了 X.25 分组交换网络，十室承担了该项目
  - 国内最早开始与国外合作从事的网络研究项目
  - 开发了 ISO X.25 底层软件，网络通信和管理软件
- 1989: NCFC
- 1992 年 6 月开始，由十室研究中国域名体系
- 1995 年 3 月，成立计算机网络信息中心

![](./20201218/1.jpeg)

谢老师给学生的一些学习建议：

> 忘记分数
>
> 独立思维
> We reject kings, presidents and voting. We believe in: rough consensus and running code.
> –– David D. Clark
>
> 动手实践
> Talk is cheap. Show me the code.
> –– Linus Torvalds

## 第二讲 网络基础：网络模型与直连网络

武老师是计算所的副研究员，研究方向是互联网体系结构和互联网测量与优化。

第二讲的主要内容是计算机网络体系结构模型和直连网络（Direct Link Networks）。

分层网络模型（Layered Network Model）：模块化方案、两层结构、三层结构、分层模型。

![](./20201218/5.png)

细腰结构（narrow-waist）是互联网体系结构模型中最典型的特征。研究表明，分层的体系结构最终会演化成细腰模型，互联网体系结构一直在演进中，现有结构可能会演化成新的细腰模型。

直连网络性能指标：带宽（Bandwidth）和时延（Latency）= 传播时延+处理时延+排队时延。

数据帧封装：添加首部和尾部界定帧的范围。

引入转义字符进行透明传输。

差错检测：奇偶检验、检验和 checksum、循环冗余校验（Cyclic Redundancy Check, CRC）。CRC 的本质是 Hash 函数。

可靠传输基本思想：确认（acknowledgment, ACK）ACK 帧、超时（timeout）重传。

停等（Stop-and-Wait）协议：最简单的可靠传输协议。

提升传输速率：序列号（Seq）

### 滑动窗口

**滑动窗口算法（Sliding-window）** ：可靠传输、高效传输、按序到达、流控功能。

- 接收端：对于每个新到达的数据帧 Seq：如果 LastFrame < Seq <= ExpMaxFrame，则接受；否则，丢弃。接受数据帧后，将收到的最大连续数据帧 Seq 作为 ACK 回复。
- 发送端：收到新的 ACK，更新 LastACK，如果窗口允许，发送新的数据帧，更新 LastFrame。

回退 N 机制（Go-Back-N）恢复丢包。只需要保证 MaxSeq/2 >= Wnd，就可以准确区分已接收和等待确认的数据帧。

![](./20201218/6.jpeg)

### 多路复用

多路复用技术：频分复用（Frequency Division Multiplexing, FDM）、时分复用（Time Division Multiplexing, TDM）、统计时分复用（Statistic TDM, STDM）、码分复用（Code Division Multiplexing, CDM）。

载波帧听多路访问（Carrier Sense Multiple Access, CSMA）

带碰撞检测（Collision Detection）的 CSMA (CSMA/CD)，用于 Ethernet。

带碰撞避免（Collision Avoidance）的 CSMA (CSMA/CA)，用于 无线局域网络，例如 WiFi。

### 以太网

以太网：MAC 地址，以太网基本上统治了有线局域网。

## WiFi

![](./20201218/7.jpeg)

接入 WiFi 热点：

- 扫描：Probe 和 Probe Response 帧
- 关联：Association Request 和 Association Response 帧
- IP 地址分配：DHCP
- 认证

### 蜂窝通信

1G 蜂窝通信是为语音通信设计的**模拟 FDM 系统**，几乎不支持数据传输。

蜂窝通信网络是覆盖范围最广的通信机制之一。

### 5G

毫米波技术、信道编码技术、大规模 MIMO、海量连接、低延迟技术、网络切片。

## 第三讲 网络互连

第二讲到第六讲，都由中国科学院计算技术研究所网络技术研究中心的武老师进行讲授。

第三讲主要讲交换网络、网络互连和数据包队列。

交换网络的设计目标是数据只朝着目的节点方向传送（转发，Forward）。

数据帧转发：给定一个包含源目的 MAC 地址的数据帧，如何确定从哪个端口转出? 

- 交换机存储目的 MAC 地址到（出）端口的映射关系（Forwarding Database, FDB）。
- 对于每个数据帧，在 FDB 中查找目的 MAC 地址对应的端口号进行单播或广播。
- 老化机制（Aging）更新 FDB。
- 每收一个新的数据帧，记录其源 MAC 地址和入端口，将该映射关系写入 FDB 表。

生成树（Spanning Tree）消除广播风暴。

IPv4

![](./20201218/8.jpeg)

无类别域间路由（Classless Inter-Domain Routing, CIDR）：前缀（prefix）长度、网络掩码（network mask）。CIDR 可更加充分的使用 IP 地址。

IP 数据包头部格式：

- Length：IP 数据包长度，最大为 65535 字节
- Protocol：标识所承载协议类型，例如 TCP: 6, UDP: 17
- Source & Destination Address：源目的 IP 地址

![](./20201218/9.jpeg)

IP 报文转发：路由器讲转发信息存储在转发表中（Forwarding Information Base），网络号和下一跳。

地址解析协议（Address Resolution Protocol, ARP）：知道下一跳 IP 地址，查询其 MAC 地址。ARP 只作用于局域网。

IP 分片（Fragmentation）：最大传输单元（Maximum Transmission Unit, MTU）。

互联网控制消息协议（Internet Control Message Protocol, ICMP）：通过发送错误代码、控制信息等来诊断和控制网络。

NAT（Network Address Translation）

IPv6：128 位。

以太网地址转换为 IPv6 地址：

![](./20201218/10.jpeg)

IPv4 地址到 IPv6 地址的映射

![](./20201218/11.jpeg)

---

2020.12.18 21:20: 这么复习挺浪费时间的。这篇复习笔记写了 2 个小时，我才看到第三个课件的一半。还有十多个课件呢，每个课件差不多 100 页，再做笔记的话没有那个时间了，还有 7 天的复习时间。

✅ 下个学期边学边做这样的笔记！吸取教训。