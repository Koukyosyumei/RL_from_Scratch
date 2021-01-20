## Re-enforcement-learning
強化学習の各種手法をスクラッチから実装していく    
言語はpython     
環境は justheuristic/practical_rl:latest を使用   

### Crossentropy
cross_entropy method でTaxi-v2を解く    

    policy[s,a] = P(take action a | in state s)  

### Deep_crossentropy
Deep_crossentropy で MountainCar-v0 を解く  

簡単に説明すると、statesから発生したactionの確率が最大になるように     
ニューラルネットワークを学習    
--> そのニューラルネットワークで、すべてのアクションの発生確率を予測する   

### Model_free  
CartPole_v0_train_q_learning.py  
  q_learnig でCartPole_v0を解く  

CliffWalking_train_sarsa_q.py  
  q_learning と sarsa で Cloffwalking を解き比較する  

### Approx  
teach a __tensorflow__ neural network to do Q-learning.


![img](https://raw.githubusercontent.com/yandexdataschool/Practical_RL/master/yet_another_week/_resource/qlearning_scheme.png)

### TODO
ハイパーパラメータをきちんと引数にしていないので改善    
