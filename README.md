# Unity-ml-agent
Using unity ml-agent with some RL algorithms implemented by pytorch


Following is kind like tutoiral or my own note while i try to understand the procedure of some parts of how ml-agent work.

# Unity ML_Agent

如果按照教學完成第一章，而在開啟上因unity版本不同，可能沒有install的選項，但是可以直接從project去到ml-agent/ml-agent中有一個unitysdk選此資料夾就可以了
<font color = 'red'>使用版本為0.4.b</font>
這版本資訊也可以從下載的github code中git log看到如下
![](https://i.imgur.com/5VjOROI.png)

<span>這裡注意此mlagent的版本不斷在改變，特別是在最新版的檔案中有些檔案似乎沒有同步好，導置在instruction上會有問題，而這裡也同樣發現github上是有一個版本選擇的，可以透過它選回以前的版本，如下圖</span>
![](https://i.imgur.com/XpjLfSU.png)

那這裡我們開始看如何使用ml_agent，首先要知道首先是創建一個新的環境，起初和一般相同都是開unity在創一個project，而這裡請把在你下載的版本中將"ML_Agents"移到你新創的project的Asset下。

這裡的ML_Agents裡頭包含"Editor"，"Examples","Plugins","Scripts"，而Examples就包含我們以前看過的environment，而plugins是tensorflowsharp執行完後有的。

那接著我們可以開始創一些需要的物件。

首先除了是一般需要的object外，我們先看ML_Agent需要那些objects，
1. Academy :是用以控制之後程式在訓練時，或是在測試時的視窗狀況，通常就放在一個empty gameobject中就可以了，其中它的Script要繼承ML_Agents中的Academy class。如下圖![](https://i.imgur.com/CCEmbRF.png)
   不需要特別override function。
2. Brain :是描述當下環境的類別和數目，而通常在程式中可能會有多個相同的環境，就是用以同時產生資料的，而這時在python中就會有多個brain，雖然一般狀況下這些brain的參數通常都是相同的。
<span style = 'color:red'>Brain通常不需要特別去override，而它也是創一個empty object去hold ML_Agent的Brain Script這裡不需要特別去Override，只要把ML_Agent/Scripts/Brain拉進去就好</span>

3. Agent :這是裡面最重要的部分，是用以執行得到的action與收集observation的，和給予reward的
   這通常需要去繼承ML_Agents的Agent，並且override幾個function，如下:
   1. Start :這滿正常的，因為要收集observation和執行動作，所以一定要初始化一些需要的變數
    
   2. AgentReset :在這function中會定義在那些情形下去重置整個環境(當然判斷和重置是都是自己要寫的)
   
   前兩個functions都不需要用到parent Agent的function。
    
   3. CollectObservations :這function中會定義到底一個環境中它回饋的observation到底是甚麼，而這裡會需要使用Agent本身的function。
   如下: AddVectorObs(float)
   AddVectorObs(0.5)這樣你之後取得的observation就會是一個(num_brain,1)的list。
   如果是更高維的，那就在其中呼叫更多得AddVectorObs，不過也可以一次放一個Vector3進去，不一定只限定一個float，但在base agent中都會將其變成多次的AddVectorObs(float)。

   4. AgentAction(float[] vectorAction, string textAction):
      第一次看到時可能會覺得怪怪的，怎麼執行的action會是Vector形式的，這裡你可以想像一下一般discrete Action的確可以使用一個值就表示這個Action會做的事情，但是在continuous中，你的動作可能一次包含著多個小動作，例如是往右1.2，往前0.8，這樣，那這時和就一定需要將這些數值都傳進來，不能像discrete定義說一個動作是如Action6 = 往右1格，往前1格這樣。
      
      而這裡其實執行動作不就是在改變unity object的transform或是其運動方式，而這function中也會去定義，這次動作的reward，而需要將reward傳出來，所以需要用Agent的function，AddReward(reward_value)，
      
      另一方面是Done，在這function中也需要去判斷

<span style = 'color:red'>這裡出現嚴重問題，在AgentAction中它的reward是在下一個state時才會算出來，而在已經over的狀況下它也還會有下一個一樣的state，而這代表我們需要將整個訓練的reward往前一位，而真正看到的state不會包含最後一個</span>

也就是如果你的寫法如下:
state_m = []
action_m = [] 
reward_m = [] 
state_m_ = [] 

while not env_info.done[0]:
    ...
    state_m.append()
    action_m.append()
    reward_m.append()
    state_m_.append()
    
learn(state_m[:-1],action_m[:-1],reward_m[1:],state_m[:-1])，其實就是需要移除錯的部分。

感覺就好像env_info得到的state是current_state沒錯，但是他的reward和local_done卻是代表上一個state的action的


所以這讓他在當下next state已經是done時，local_done仍是False，所以又在進入，但是這不對，當然這可能需要再測試

這裡補充一下使用上似乎無法一次用不同的unity環境，而且不正常的關閉，會使之後程式無法開啟unity的環境。


使用說明，這裡我們先探討在python上程式是要如何執行，首先當然是要有github的ml_agents的檔案，其中有一個ml_agents/python/env，而這裡面會放我們build好的環境，假如已經build環境這時就可以創建一個jupyter notebook，使用上就是

```python=
from unityagents import UnityEnvironment
```
這樣就能使用unity ml_agent的程式了。

```python=
env_name = 'env/rollerball/rollerball'  
env = UnityEnvironment(file_name = env_name)
```
env_name指的是該exe檔的名字。

而此部分"env"就如同openai gym有同樣的功能了

那下面介紹幾個該從env拿取的東西

1. brain
2. brain_name

等等我們透過一張圖來解釋整個ml agent是如何溝通的

![](https://i.imgur.com/LFVPFem.png)

我們需要知道env中的brain，與如何選擇是哪個

```python=
brains = env.brains #dict
brain_names = env.brain_names # list
select_brain = brains[brain_names[0]] # select_first brain
```
而我們可以透過brain_names取得想要得brain了

```python=
dim_obs = select_brain.vector_observation_space_size
dim_act = select_brain.vector_action_space_size
```
這時model也準備好的話，要和環境互動就要選擇該model使用的brain。
每次和環境互動完都會獲得一個dict，就稱env_info好了，可以用他拿取
```python=
env_info = env.reset(train_mode = False)[brain_names[0]]
state = env_info.vector_observations  
state = torch.tensor(state).type(torch.FloatTensor).to(device)
action = model.predict(state)

env_info = env.step(action.detach().cpu().numpy())[brain_names[0]]
```
只要是和環境有互動的(step和reset)就需要額外給予brain_name。

而主要python程式就這樣。

