"""One module per family. Every ``learn(cfg, agent, rec)`` trains ``agent`` in place and
reports to the shared ``Recorder`` (see ``cartpole_rl.metrics``)."""

from cartpole_rl.algorithms import actor_critic, dqn, policy_gradient, ppo, tabular

LEARNERS = {
    "q-learning": tabular.learn,
    "sarsa": tabular.learn,
    "mc-control": tabular.learn,
    "dqn": dqn.learn,
    "double-dqn": dqn.learn,
    "dueling-dqn": dqn.learn,
    "reinforce": policy_gradient.learn,
    "reinforce-baseline": policy_gradient.learn,
    "actor-critic": actor_critic.learn_one_step,
    "a2c": actor_critic.learn_a2c,
    "ppo": ppo.learn,
}
