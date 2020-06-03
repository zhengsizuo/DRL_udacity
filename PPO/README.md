# Proximal Policy Optimization
## Code explanation 
- **ac_model.py:** build separated Actor-Critic architecture
- **ppo.py:** trained on the toy example Pendulum, set LOAD_MODEL as True to watch the intelligent agent
- **ppo_bipedalwalker.py:** trained on BipedalWalker-v2  
- **load the tensorboard:** tensorboard --log_dir = runs

## Results
### rewards figure
<table>
    <tr>
        <td ><center><img src="figures/pen_scores" width="300">Fig.1 pendulum rewards</center></td>
        <td ><center><img src="figures/bipedal_scores.png"  width="300">Fig.2 PG with baseline</center></td>
    </tr>
</table>

### intelligent agent
Test rewards of Pendulum agent in ten episodes:
> Average rewards:-230.52313164492534   Average steps:200.0 

![best](figures/ppo_pendulum.gif) <br />

Test rewards of BipedalWalker agent in ten episodes:
> Average rewards: 239.6173305411697    Average steps: 1181.4

![best](figures/ppo_bipedal.gif) <br />