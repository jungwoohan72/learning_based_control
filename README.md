# learning_based_control
ME491 Term Project - Control Husky via deep Reinforcement Learning

Goal of this project is to control the husky from random initial points toward the designated destination (center of the envrionment) as quickly as possible.

Allowed information is:  
1. pose, linear velocities and angular velocities w.r.t the world frame
2. Wheel speed
3. 20 LiDAR points with respective to the robot body frame.

Used algorithms are PPO and SAC. 

Reward shaping, curriculum learning, and hyperparameter tuning are used to improve the performance.

The detailed result is in the report.pdf.

![Short](https://user-images.githubusercontent.com/45442859/145036329-4c304298-eef6-4512-862f-e8c4a4cf16a2.gif)
![Middle](https://user-images.githubusercontent.com/45442859/145036426-925de79f-f9bd-483e-9aca-01b6cbbfad76.gif)
![Long](https://user-images.githubusercontent.com/45442859/145036659-1cf720d3-c27e-4d6c-b375-5a5674aa26d4.gif)
