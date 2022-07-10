# 强化学习-学习笔记x | 环境配置 

## 1. OpenAI Gym介绍

OpenAI Gym https://gym.openai.com  是强化学习最常用的标准库。如果得到了 $\pi$ 函数或者 $Q*$ 函数，就可以用于Gym的控制问题和小游戏，来测试算法的优劣。

按照官方文档，安装 gym，就可以用 python 调用 gym 的x函数。

### 1.1 简易安装

简易安装过程：

```python
pip install gym==0.15.7 -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
```

截止2022-07-03，gym最新版本是0.21，但由于我的 python 环境为 3.6.4，所以我的 gym版本需要下降。

> 我的 pip 最近总是连接不到远端库，执行`pip install 库`会报错：
>
> ```python
> Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None))
> after connection broken by 'SSLError(SSLError(1, u'[SSL: CERTIFICATE_VERIFY_FAI
> LED] certificate verify failed (_ssl.c:726)'),)': /packages/0f/fb/6aecd2c8c9d0ac
> 83d789eaf9f9ec052dd61dd5aea2b47ffa4704175d7a2a/psutil-5.4.8-cp27-none-win_amd64.
> whl
> ```
>
> [python使用pip安装模块出错 Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) - Aimed - 博客园 (cnblogs.com)](https://www.cnblogs.com/aimed/p/10178048.html)
>
> 所以需要命令行后面的部分。



安装之后测试一下是否成功：

```python
import gym
 
test_envs={'algorithm':'Copy-v0',
           'toy_text':'FrozenLake-v0',
           'control':'CartPole-v0',  
           'atari':'SpaceInvaders-v0'
           'mujoco':'Humanoid-v1',    
           'box2d':'LunarLander-v2' } 
 
game_name = test_envs['algorithm']
env = gym.make(game_name)
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render() 
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
```

没有问题，表示 gym 包安装成功。



pygame：

```python
pip install pygame -i http://mirrors
.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
```



安装box-2d 出错

```python
pip install box2d-py -i http://mirro
rs.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
```

应当前往Python Extension Packages for Windows下载相应的包：[AttributeError: module ‘gym.envs.box2d‘ has no attribute ‘LunarLander‘_](https://blog.csdn.net/weixin_44480968/article/details/124743355)

接下来安装 Atari ，

```python
pip install atari_py -i http://pypi.
douban.com/simple --trusted-host pypi.douban.com
```

然后其他的东西我还没有装。如果装了会更新在安装笔记里。参考教程为：

1. https://blog.csdn.net/weixin_33654339/article/details/113538141
2. [Gym Documentation (gymlibrary.ml)](https://www.gymlibrary.ml/)
3. https://github.com/openai/gym



### 1.2 gym 简单例程1

```python
import gym
import time
# 生成 CartPole 环境
env = gym.make('CartPole-v0')

# 重置环境
state = env.reset()

# 这里的循环就是上面 s,a,r的过程
for t in range(10000):
    # 弹出窗口来显示游戏情况
    env.render()
    print(state)

    # 随机均匀抽样一个动作记为 action
    # 这里是为了图方便，实际应用应该通过policy函数或者Q*来选择。
    action = env.action_space.sample()
    
    # 把action输入到step()函数,即agent执行这个动作
    state,reward,done,info = env.step(action)
    # 为了不让窗口太快消失
    time.sleep(1)
    if done:
        print("Finished")
        break


env.close()
```

运行效果：

![](https://img2022.cnblogs.com/blog/2192866/202207/2192866-20220704101217447-1831399251.png)
![](https://img2022.cnblogs.com/blog/2192866/202207/2192866-20220704101224260-2126668986.png)



虽然例程运行的很好，但总感觉缺少了点什么，可能是安装的太容易，跟深度学习比较起来步骤比较少？

目前是例程可以跑起来，后续缺少东西会继续安装。

## 2. 一些参考资料/

[强化学习入门：环境（含机器人）和代码库介绍 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/477982098)

[强化学习基础篇（十）OpenAI Gym环境汇总 - 简书 (jianshu.com)](https://www.jianshu.com/p/e7235f8af25e)

[专用于机械臂强化学习的Gym库]([qgallouedec/panda-gym: OpenaAI Gym Franka Emika Panda robot environment based on PyBullet. (github.com)](https://github.com/qgallouedec/panda-gym/))