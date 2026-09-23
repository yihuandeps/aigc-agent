<!-- 真实感与 UGC 类模板参考。生成自 skills/seedance-prompting/data/（上游 LearnPrompt/awesome-seedance，策划内容 CC BY 4.0，案例出自 goodcase.ai），2026-09-23 由 scripts/build_seedance_skill.py 生成 —— 勿手改，改脚本重跑 -->


# 真实感与 UGC（4 个模板）

靠写相机缺陷、身体损耗和消费级镜头换真实感的模板，不靠堆画质词。

目录：手持 UGC vlog · 第一人称一镜到底 · 早年 DV 家庭录像 · 宠物动物当主角

## 手持 UGC vlog

用相机缺陷换真实感。指名一个具体的消费级器材年代，把它的毛病写成要求，再手动关掉电影感。

**适用场景：** 要私人感的素材：日常、旅拍、健身、做饭、出门前准备。目标是像真有人拍的，而不是像很贵的时候用这套。

**结构（按块填，别跳）：**
1. CAMERA：机器怎么拿、什么年代、有哪些操作毛病
2. LOOK：磁带或胶片质感、颗粒、光晕、对比度、曝光行为
3. STYLE：节奏和情绪，一两行写完
4. SUBJECT 与 SETTING：谁、在哪，都写短
5. STORYBOARD：`→ (3s, propped medium shot)` 这种短行，配一句口语台词
6. AUDIO NOTES 与 REALISM NOTES：环境音清单，然后是肢体语言和瑕疵清单

**要点（来自真实生效的案例）：**
- 把相机缺陷当真实感开关：手抖、对焦来回找、曝光呼吸、构图漂移、变焦不匀、偶尔切掉半张脸。案例库里 23 条靠这套词表拿到手机实拍质感。
- 指名器材年代，不要笼统要求真实：mini DV 家用摄像机、16mm、VHS、iPhone 16 Pro、胸挂运动相机。一个具体型号带着整套光学特征，realistic 这个词带不来。
- 显式关掉电影感：no cinematic emulation、不使用稳定器、不做电影式运镜、无美颜、无磨皮。
- 写单调递进的身体状态，给模型一个不可逆的时间线索。骑行 vlog 那条写死汗量随时间递增不可倒退，并逐镜从额角第一层汗写到骑行服全湿。
- 台词跟着分镜行走，不要单开对白块，让说话和动作焊在一起。

**常见坑：**
- 同一条 prompt 里既要手持真实感又要 4K 电影级打光。两套光线逻辑打架，结果落在塑料感上。
- 让景别推到大特写。男友视角那条明确禁止脸部填满画面，最近只给到胸口以上，因为大特写会暴露 AI 脸。
- 用数字变焦当转场。要一镜到底就补一句禁止数字变焦、突然推近和隐形剪辑。
- 台词写太长。长句抢画面还拖垮口型，每句控制在八个词以内。

**可复制引导语（把【】里的换成你的内容，连同下面这段模板一起发给模型）：**

> 我要做一条手持感的生活 vlog，【出镜的人是：一位二十多岁的女生，穿宽松卫衣】，【场景是：周末早上在自家厨房做手冲咖啡】，有参考图我会一起发给你。请根据下面这个提示语模板，帮我改写成一条可以直接用的 Seedance 视频提示语：

**锚定案例（按热度）：**
- [mini DV 咖啡 ASMR vlog](https://goodcase.ai/cases/seedance-25-minidv-coffee-asmr-vlog)　热度 99，稳定 77　Seedance 2.5
- [Seedance 原生 UGC 竖屏手机跟拍短片](https://goodcase.ai/cases/mightyking-seedance-ai-7bbc1d4f9ad9)　热度 95，稳定 71　Seedance
- [Seedance 2.5 印尼女生日常写实短片](https://goodcase.ai/cases/rishuavr-seedance-ai-ad4e6de3949d)　热度 95，稳定 78　Seedance

## 第一人称一镜到底

执法记录仪、GoPro、FPV 和车把视角。相机挂在身体上，运动必须从身体推导，每一次剪辑都得手动声明。

**适用场景：** 要观众就是操作者的沉浸素材：破门突入、极限运动、厨师视角做饭、无人机飞行。207 条里 32 条属于这类。

**结构（按块填，别跳）：**
1. SCENE CONTEXT：一段话交代主体、挂载方式和总时长
2. ACTIVE REFERENCES：场景、手和道具的命名 token
3. LOCATION MAP：每段的前景、中景、背景各是什么，以及机位高度
4. FIRST FRAME / BLOCKING：首帧非空，开场就在动作中间
5. FORMAT MODE：硬切落在哪里，哪几段是连续一镜
6. OPTICS：每段的视场角，附一句段内不许漂移
7. 时间轴与音频

**要点（来自真实生效的案例）：**
- 声明挂载位置和高度，模型才能推出该怎么晃：胸挂在破门手身上、POV 保持胸到眼的高度、只随身体移动。
- 拒绝空首帧。GoPro 钓鱼那条写 `Non-empty opening frame: already mid-cast, rod raised, line already peeling off the reel`，把死掉的第一秒省掉了。
- 一镜到底和剪辑要分开声明。剪点写成清单——A 0-9s 河边一镜，HARD CUT，B 9-21s 案板一镜——再补一句除此之外相机不剪。
- 视场角逐段写成度数（84° 在搏斗中收到 63°，下一段 63° 收到 18°），后面跟一句 `No drift within any segment`。
- 把身体挂载的光学后果写出来：边缘广角畸变、行走造成的上下颠动、快速转头的运动模糊、手电只照亮操作者面向的方向。

**常见坑：**
- 操作者自己的脸入画。补一句 `the camera itself is never visible`，只描述手在做什么。
- 手入画却不说左右手和持物。要写清哪只手拿什么，否则会长出第三只手。
- 在标了一镜到底的段落里安排跨场景大跳。要么实时走过去，要么在边界放一个声明过的硬切。
- 忘了禁掉电影化处理。执法记录仪和运动相机素材要明写 no slow-motion, no cinematic grading，否则会变成电影预告片。

**可复制引导语（把【】里的换成你的内容，连同下面这段模板一起发给模型）：**

> 我要做一条第一人称一镜到底的视频，【我的视角是：骑着山地车从林道冲下山】，【画面里要出现我的双手和车把】。请根据下面这个提示语模板，帮我改写成一条可以直接用的 Seedance 视频提示语：

**锚定案例（按热度）：**
- [Seedance 2.5 生成 GoPro 钓鱼到烤鱼全流程](https://goodcase.ai/cases/seedance-2-5-gopro-94a73eef1dbf)　热度 84，稳定 79　Seedance 2.5
- [魔法画笔将城市交通变成动漫](https://goodcase.ai/cases/seedance-magic-pen-street-transport-vlog-28d80bd05eda)　热度 81　Seedance
- [First-Person POV Dragon Rider Cinematic](https://goodcase.ai/cases/first-person-pov-dragon-rider-cinematic)　热度 75，稳定 77　Seedance 2.5

## 早年 DV 家庭录像

年代感靠机器缺陷和一件生活小事撑起来。把 DV 的对焦拉风箱、曝光跳动、手抖写成明确清单，剧情放小，整条就像一盘真的旧带子。

**适用场景：** 家庭录像、旅拍日志、MiniDV 情侣片、街区漫步，任何想看起来像多年前用家用机器拍下来的生活片段。

**结构（按块填，别跳）：**
1. 开场一句：时长、分辨率、片种写成 early-2000s DV home video，说明有没有参考图
2. MAIN SUBJECT：年龄、皮肤、发型、整套衣服，收一句一致性锁
3. SETTING：具体的生活化街区和它的杂物，末尾排除地标、广告、品牌
4. CAMERA：DV 机器的缺陷清单，再加一句禁掉稳定器和电影感运镜
5. 按时间码或小标题分段，一段一件小事，台词写进它发生的那一拍
6. AUDIO：只留现场音，写明 no music
7. 收尾：realism 约束、负面清单、画幅，最后硬切到黑

**要点（来自真实生效的案例）：**
- 相机那段写成器材缺陷清单。首尔夏日午后那条整段列 `autofocus hunting, exposure shifts, accidental zooms`，紧跟着禁掉 `No stabilization, drone footage, gimbal movement`。
- 人物一句话锁死，整条复用。首尔夏夜 Vlog 用 `Maintain the same face, hairstyle, clothing, body proportions` 收尾；有参考图就换成情侣约会那条的 `Use the uploaded reference image as the exact character reference`。
- 场景堆生活痕迹，然后把地标排除掉。首尔午后那条点名盆栽、电线杆、晾在外面的衣服，再补一句 `No tourist attractions, advertisements, recognizable brands`。
- 整条只给一件小事，别给剧情。树叶那条三十秒就是一片叶子掉到她头上，她试着把叶子立在自行车座上，两次都被风吹掉。
- 结尾用录像带式的硬切，声音只留现场。首尔午后那条跟拍她转过街角，然后 `The recording abruptly cuts to black`，音频只有脚步、虫鸣、自行车铃，写明 `No music`。

**常见坑：**
- 往里堆 4K、cinematic lighting、sharp detail。画质一上去 DV 味就没了，年代感只能靠缺陷清单换。
- 三十秒塞五件事。一个时间码槽只放一个动作加一个反应，树叶那条一件事就占满六秒。
- 道具在两拍之间消失或者变成两个。首尔午后那条专门写了球踢回去之后留在孩子那边，不会回来也不会复制，每个道具的去向都要点名。
- 台词写成长句。口型一糊就该砍句子，案例里的台词都不超过一句，像 `Okay, that was pointless.`

**可复制引导语（把【】里的换成你的内容，连同下面这段模板一起发给模型）：**

> 我要做一条年代感家庭录像风格的视频，【时间设定在 2003 年夏天，用当时的家用 DV 拍的】，【拍的是我表妹放学回家路上遇到一只猫】。请根据下面这个提示语模板，帮我改写成一条可以直接用的 Seedance 视频提示语：

**锚定案例（按热度）：**
- [首尔夏日街巷里的悠闲午后](https://goodcase.ai/cases/seedance-prompt-create-a-30-second-1080p-ultra-realistic-personal-home-video-showing-a-f4036ce777fe)　热度 99　Seedance
- [首尔夏夜 Vlog](https://goodcase.ai/cases/vlog-c8171f712492)　热度 99，稳定 76　Seedance 2.5
- [韩系情侣的街头约会日记](https://goodcase.ai/cases/seedance-use-the-uploaded-reference-image-as-the-exact-character-reference-214303ebc4cf)　热度 92　Seedance

## 宠物动物当主角

动物是真正的主角，镜头就交给一台手机。数量锁死成一只，动物只做动物做的事，包袱留给它一步步逼近镜头。

**适用场景：** 猫狗或野生动物抢镜的自拍、vlog 片段，以及靠一个真实动物行为撑起来的写实喜剧。

**结构（按块填，别跳）：**
1. 参考与主体段：有人出镜就用参考图锁住人，再写死只有一只动物，从头到尾是同一只
2. 格式段：时长、竖屏 9:16、手持前置自拍、室内自然光、不调色不加美颜
3. 相机段：手臂漂移、构图偏一点、自动对焦拉风箱、不剪、不变焦、没有第三方机位
4. 正文按秒切段，每段让动物往前递进一步：注意到、伸爪、抢走、爬肩、扑镜头
5. 人的反应和半句没说完的台词，写在同一拍里
6. 音频段：现场声清单，没有音乐、字幕、水印
7. 严格约束收尾：一个人一只动物、不许出现第二只、镜面物理正确、画面里没有摄影师

**要点（来自真实生效的案例）：**
- 数量写成一个硬数字，并把这一只写成物理连续。狗狗抢镜那条是 `Use exactly ONE small playful dog throughout the entire video`，外加 `No duplicate animal`；小猫 vlog 那条还逐项点名 `consistent fur pattern, eye color, size, whiskers, ears, paws`。
- 给动物单开一段行为规则。猕猴那条写了 ANIMAL BEHAVIOR 块，`No talking, no human clothing, no human-like walking`，笑点全押在猴子跟着徒步者歪头这件事上。
- 真实感用一份相机缺陷清单换。猫咪一日 vlog 要的是 `Natural handheld shake`、`Occasional autofocus hunting`、`Natural front-camera lens distortion`，再禁掉 `No cinematic camera movements`。
- 动作按拍升级，最后落到镜头上。狗狗那条最后一段写 `Its nose fills a large part of the frame`，画面先失焦再找回；雨天小猫那条是爪子伸到镜头角落，片子在笑声里断掉。
- 声音只留现场音。猫咪 vlog 收尾的音频段只列 meows、purring、chirping、footsteps，并写死 `No background music`；狗狗那条写的是 `Natural room ambience only`。

**常见坑：**
- 数量留着不写。片子中段会多出第二只动物或者多一个倒影，狗狗那条用 `Exactly one woman. Exactly one dog.` 和 `No duplicate reflection` 把它钉死。
- 让动物说话或者像人一样走路。片子会滑向动画，猕猴那条直接禁掉这些，把笑点压回真实的动物行为。
- 给人写完整长句台词。演员要边笑边念，口型必散，改成被笑打断的半句，像 `You little—` 这种断在一半的。
- 顺手加电影感运镜和调色。手机拍的质感立刻就没了，写上 no cinematic lighting、no color grading、no cuts、no zoom。

**可复制引导语（把【】里的换成你的内容，连同下面这段模板一起发给模型）：**

> 我要做一条宠物抢镜的手持自拍视频，【我的宠物是一只胖橘猫，右耳有个小缺口】，【它趁我拍自拍一路爬到肩膀上，最后把鼻子怼到镜头前】。请根据下面这个提示语模板，帮我改写成一条可以直接用的 Seedance 视频提示语：

**锚定案例（按热度）：**
- [狐狸在森林溪流边自拍漫游](https://goodcase.ai/cases/mrdasonx-seedance-ai-ccaa50150259)　热度 92，稳定 75　Seedance
- [猫咪自拍记录温馨的一天](https://goodcase.ai/cases/zarairahh-seedance-ai-f89372941867)　热度 89，稳定 70　Seedance
- [Cats Chasing via Red Mini Motorcycle](https://goodcase.ai/cases/cats-chasing-via-red-mini-motorcycle)　热度 84，稳定 89　Seedance 2.0
