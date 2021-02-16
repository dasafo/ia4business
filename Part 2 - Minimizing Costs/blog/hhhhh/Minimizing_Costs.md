# Reducing costs of data center cooling

We will configure our own server environment and build an AI that will control
the cooling/heating of the server to keep it in an optimal temperature range and
at the same time we need to save the maximum energy, thus minimizing costs. Our
goal will be to achieve at least 40% energy savings.

## 1 - Boundary Conditions and program operation

**- Parameters:**

* The average atmospheric temperature for a month.
* The optimal range of server temperature: **[15ºC, 25ºC]**.
* The minimum server temperature: **-25ºC** (it doesn't operate below).
* The maximum server temperature:  **80ºC** (it doesn't work above).
* The minimum number of users on the server: **8**.
* The maximum number of users on the server: **120**.
* The maximum number of users on the server that it can up or down per minute:
**6**.
* The minimum data transmission rate on the server: **20**.
* The maximum data transmission speed on the server: **400**.
* The maximum data transmission speed that can go up or down per minute: **10**.


**- Variables:**

* Server temperature at any time.
* Number of users on the server at any time.
* Data transmission speed at any minute
* AI's energy expended on the server (to cool or heat it) at any time.
* The energy expended by the server's integrated cooling system that
automatically brings the server temperature to the optimal range whenever the
server temperature is out of this optimal range.

**We are going to make two assumptions to facilitate the work:**

* First, the server temperature can be approximated by Multiple Linear
Regression:

$$Server \hspace{2mm} temp. = a_0 + b_1 \cdot atmosph. temp. + b_2 \cdot
Num.users + b_3 \cdot data\hspace{2mm} trans.rate$$

where $a_0 \in \mathbb R$, $b_1,b_2,b_3 > 0$

When the atmospheric temperature rieses, the server temperature rises too. The
more users are active on the server, the more the server will spend to handle
them, and therefore the server temperature will be higher. And finally, the more
data is transmitted within the server, the more the server will spend to process
it, and therefore the higher the server temperature will be. And for simplicity,
we just assume these correlations are linear.

We are going to give values to parameters:

$$Server \hspace{2mm} temp. = 0 + 1 \cdot atmosph. temp. + 1.3 \cdot Num.users +
1.3 \cdot data\hspace{2mm} trans.rate$$

* Second, the energy expended by our system (IA or IRS) that changes the server
temperature from $T_t$ to $T_{t+1}$ in a unit of time (1 minute in our problem),
it can be approximated again by linear function regression of the absolute
change in server temperature:

$$E_t=\alpha |\varDelta T_t| + \beta = \alpha |T_{t+1} - T_t| + \beta$$

Where:

- $E_t$ is the energy expended by the system between $t$ and $t+1$.
- $\varDelta T_t$ is the change of temperature of the system between $t$ and
$t+1$.
- $T_t$ and $T_{t+1}$ are the server temperature in $t$ and $t+1$.
- $\alpha >0$ and $\beta \in \mathbb R$. Althought we put them like $\alpha=1$
and $\beta=0$(for example).

Finnaly we have:
$$
\begin{equation*}
E_t = |\Delta T_t| = |T_{t+1} - T_t| =
\begin{cases}
T_{t+1} - T_t & \textrm{si $T_{t+1} > T_t$, (the server heats up) } \\
T_t - T_{t+1} & \textrm{si $T_{t+1} < T_t$ (the server gets cold) }
\end{cases}
\end{equation*}
$$

### How the simulation works:

We are going to simulate a real server and for that the number of users and data
transmission speed will fluctuate randomly. AI has to understand how much
cooling and heating power it must transfer to server. In this way the program
must be able to spend the least energy optimizing its heat transfer.

### General operation:

Two possible systems can regulate the server temperature: the AI or the server's
integrated cooling system. The integrated server cooling system is a non-
intelligent system. This system will automatically return the server temperature
to its optimal temperature.
The integrated system(no-IA) works only when IA system not work. If IA is
enabled, no-IA is disabled and then the IA updates the server temperature. The
IA system is non-deterministic because it predicts what temperatureto set based
on the previous parameters.

The IA must spend less energy than the intern cooling system. Therefore we can
see in the second assumption the energy is  proportional to temperature.

The energy saved by IA every time (1 minute) is:
$$
\begin{align*}
        \textrm{IA Energy saved between $t$ y $t+1$}
        & = |\Delta T_t^{\textrm{Intern System Server (no-IA)}}| - |\Delta
T_t^{\textrm{IA}}| \\
        & = |\Delta T_t^{\textrm{no IA}}| - |\Delta T_t^{\textrm{IA}}|
\end{align*}
$$

Our goal is to save the maximum energy every minute, thus save the maximum total
energy for 1 whole simulation year, and finally save the maximum costs on the
cooling/heating electricity bill.


Now let's to define the states, actions and rewards.
*(You can get more information about Q-learning from here: [A Beginners Guide to
Q-Learning](https://towardsdatascience.com/a-beginners-guide-to-q-
learning-c3e2a30a653c)*

### States

The state $s_t$ is a vector that made up of:
* Server temperature in time *t*.
* Number users in *t*.
* Data transmission speed in *t*.

### Actions

- **0**: IA cools the server 3ºC *(- 3ºC)*.

- **1**: IA cools the server 1.5ºC *(- 1.5ºC)*.

- **2**: IA makes no difference.

- **3**: IA heats up the server 1.5ºC *(+ 1.5ºC)*.

- **4**: IA heats up the server 3ºC *(+ 3ºC)*.


### Rewards

The reward is the energy expended on the server that the AI is saving relative
to the server's intern cooling system (no-IA):

$$
\textrm{Reward}_t = E_t^{\textrm{no IA}} - E_t^{\textrm{IA}}
$$

Then:

$$
\begin{align*}
    \textrm{Reward}_t
    & = \textrm{Saved energy by IA between $t$ and $t+1$} \\
    & = E_t^{\textrm{no IA}} - E_t^{\textrm{IA}} = |\Delta T_t^{\textrm{no IA}}|
- |\Delta T_t^{\textrm{IA}}|
\end{align*}
$$

## 2 - Deep Q-Learning algorithm

For each state $s_t$:
* $Q(s_t,a_t)$ is the prediction and we choose $a_t$ with *argmax* or *softmax*
([https://deepai.org/machine-learning-glossary-and-terms/softmax-
layer](https://deepai.org/machine-learning-glossary-and-terms/softmax-layer)).
* The target value is:
$$
r_t + \gamma \underset{a}{\max}(Q(s_{t+1}, a))
$$
, where $r_t=R(s_t,a_t) + \gamma \underset{a}{\max}(Q(s_{t+1},a))$ is the
reward, and $\gamma$ is the discount factor
([https://en.wikipedia.org/wiki/Q-learning#Discount_factor](https://en.wikipedia.org/wiki/Q-learning#Discount_factor))
* Loss error between the prediction and the target is:
$$
\textrm{Loss} = \frac{1}{2} \left( r_t + \gamma \underset{a}{\max}(Q(s_{t+1},
a)) - Q(s_t, a_t) \right)^2 = \frac{1}{2} TD_t(s_t, a_t)^2.
$$
, where $TD_t(s_t,a_t) = R(s_t,a_t) + \gamma \underset{a}{\max}(Q(s_{t+1},a)) -
Q(s_t,a_t)$.
This loss error spreads to net back and weights are update.

Other problem to solve, $s_t$ is almost always highly correlated with $s_{t+1}$,
therefore our system isn't learning as much as we would like. Let's consider the
last *m* transitions blocks (*m* is a big number). This is known as **Experience
Replay**.

Our brain will have 64 neurons in the first layer and 32 neurons in second
layer. This neural network takes the environment's states as inputs and returns
the Q-values for each of the 5 actions as outputs. This artificial brain will be
trained with a "mean square error" loss *(MSE)* and an *Adam optimizer*.

<img src="Recursos/er_neuronal.jpg" width="800">

### Algorithm steps:

First we need to start the Expereince Replay like a empty list **M**. Then:

**1 -** $Q(s_t)$ prediction.

**2 -** We execute the maximum Q-values acttion:
$a_t = \underset{a}{\textrm{argmax}} Q(s_t, a)$.

**3 -** We get the Reward: $r_t = E_t^{\textrm{no IA}} - E_t^{\textrm{IA}}$.

**4 -** Next state $s_{t+1}$.

**5 -** Update Experience Replay memory *(M)* with $(s_t, a_t, r_t, s_{t+1})$.

**6 -** We select a random block transitions $B \subset M$ $(s_{t_B}, a_{t_B},
r_{t_B}, s_{t_B+1})$, then we get the new predictions, new targets and the new
loss error. Finally the error spreads throw the neuronal network with a
Stochastic gradient descent and the weights will be update.

### Algorithm implementation:

**1 -** Building the environment.

**2 -** Builiding the brain.

**3 -** Deep Reinforcement Learning algorithm.

**4 -** Training the IA.

**5 -** Testing the IA.
