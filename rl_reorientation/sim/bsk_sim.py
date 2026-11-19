import numpy as np
from Basilisk.architecture import messaging
from Basilisk.simulation import spacecraft, reactionWheelStateEffector, extForceTorque, ExtPulsedTorque
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeRW, simHelpers
from Basilisk.utilities import RigidBodyKinematics as rbk

simProcessName = "simProcess"
simTaskName = "simTask"
inertia = np.array([0.02 / 3, 0.1256 / 3, 0.1256 / 3]) # inertia of the satellite
U_MAX = 0.001   # max inertia of the reaction wheels

# tune based on desired performance and accuracy. The agent updates its commands each control step
# but the physics are simulated controlTimeStep/simTimeStep times per control step
simTimeStep = 0.1
controlTimeStep = 0.5


# ------------------------------------------------------------------------
# Attitude conversion helpers: agent uses quaternions, Basilisk uses MRPs
# ------------------------------------------------------------------------
def quat_normalize(q):
    """Make q unit length and pick the sign with w >= 0.
    q and -q are the same rotation"""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0 else q

def quat_to_mrp(q):
    """Quaternion to MRP"""
    return rbk.MRPswitch(np.array(rbk.EP2MRP(quat_normalize(q))), 1.0)  # EP = Euler parameters = quaternion

def mrp_to_quat(mrp):
    """MRP to quaternion (with w >= 0)"""
    return quat_normalize(np.array(rbk.MRP2EP(np.asarray(mrp, float))))

def sample_moi(rng, inertia, moi_uncertainty=0.2, max_tries=1000):
    """Samples disturbed inertia vector that is triangularly valid (-> makes sense physically)"""
    for _ in range(max_tries):
        inertia_dist = inertia * rng.uniform(1 - moi_uncertainty, 1 + moi_uncertainty, size=inertia.shape)
        a, b, c = inertia_dist
        if a + b >=c and a + c >= b and b + c >= a:
            return inertia_dist
    raise RuntimeError("no valid inertia found")


class SatelliteSim:
    def __init__(self, initial_att, target_att, dist_torque_static, dist_torque_pulsed, moi_uncertainty):
        """
        :param initial_att:  quaternion
        :param target_att:   quaternion
        :param dist_torque_static:  external disturbance torque that is constantly applied to the satellite body [x,y,z]
        :param moi_uncertainty:  uncertainty of the moment of inertia
        """
        self.n_physics = int(round(simTimeStep / controlTimeStep))  # physics steps per control step
        assert abs(self.n_physics * simTimeStep - controlTimeStep) < 1e-9, "controlTimeStep must be a multiple of simTimeStep"
        self.t = 0.0    # simulated time [s]
        self.targetAtt = quat_to_mrp(quat_normalize(target_att)) # target attitude stored as MRP to be used by the PD controller

        # -------------------------
        # 0 - Simulation Setup
        # -------------------------
        # create simulation module
        self.scSim = SimulationBaseClass.SimBaseClass()
        proc = self.scSim.CreateNewProcess(simProcessName)
        # create the dynamics task
        proc.addTask(self.scSim.CreateNewTask(simTaskName, macros.sec2nano(simTimeStep)))


        # -------------------------
        # 1 - Initialize Satellite
        # -------------------------
        self.satellite = spacecraft.Spacecraft()
        self.satellite.ModelTag = "satellite"
        self.satellite.hub.mHub = 4.0  # mass of 4kg
        self.satellite.hub.r_BcB_B = [[0.0], [0.0], [0.0]]  # center of mass is at the body origin
        self.satellite.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(np.diag(inertia)) # inertia matrix of the hub, taken about the hub's center of mass (Bc), expressed in the body frame B
        self.satellite.hub.sigma_BNInit = quat_to_mrp(initial_att)
        self.scSim.AddModelToTask(simTaskName, self.satellite, ModelPriority=1)


        # -------------------------
        # 2 - Reaction Wheels
        # -------------------------
        # reaction wheel factory
        self.rwFactory = simIncludeRW.rwFactory()
        for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):  # create RW for each axis
            self.rwFactory.create("custom", axis, Omega=0.0,    # start with the wheel not spinning
                                  Omega_max=6000.0,     # speed limit
                                  maxMomentum=0.01,     # momentum limit
                                  u_max=U_MAX,          # motor torque limit
                                  RWModel=messaging.BalancedWheels)

        # create RW object container and tie to our satellite
        self.rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
        self.rwStateEffector.ModelTag = "RW_cluster"
        self.rwFactory.addToSpacecraft(self.satellite.ModelTag, self.rwStateEffector, self.satellite)  # adds the RW container to the satellite
        # add the RWs to the simulation process (otherwise it would not be part of the update cycle and the logs do not work)
        self.scSim.AddModelToTask(simTaskName, self.rwStateEffector, ModelPriority=2)

        # message through which the agent sends the RW torque commands
        # TODO: make the agents input cmd independent of RW wheel layout (by using basilisk module)
        self.cmdMsg = messaging.ArrayMotorTorqueMsg()
        self.rwStateEffector.rwMotorCmdInMsg.subscribeTo(self.cmdMsg)


        # -------------------------
        # 3 - External Disturbance
        # -------------------------
        # external disturbance torque/force module
        self.extFT = extForceTorque.ExtForceTorque()
        self.extFT.ModelTag = "extForceTorque"
        self.satellite.addDynamicEffector(self.extFT)
        self.scSim.AddModelToTask(simTaskName, self.extFT, ModelPriority=3)
        # external torque static
        self.torqueMsg = messaging.CmdTorqueBodyMsg()
        self.extFT.cmdTorqueInMsg.subscribeTo(self.torqueMsg)
        # TODO: add the pulsed disturbance torque

        # self.extFT.extTorquePntB_B = list(map(float, dist_torque_static))
        # external torque periodic
        # self.extFT_periodic = ExtPulsedTorque.ExtPulsedTorque()
        # self.extFT_periodic.ModelTag = "extPulsedTorque"


        # -------------------------
        # 4 - Initialize Simulation
        # -------------------------
        # initialize all modules; after that the simulation is ready to run
        self.scSim.InitializeSimulation()
        self._write_cmd(np.zeros(3))
        self._write_dist(dist_torque_static)


    def _write_cmd(self, torques):
        """Write the wheel motor torques (3 values x,y,z in Nm) into cmdMsg"""
        payload = messaging.ArrayMotorTorqueMsgPayload()
        # the message has room for many wheels. Only fill in the first three and zero the rest
        payload.motorTorque = list(map(float, torques)) + [0.0] * (len(payload.motorTorque) - 3)
        self.cmdMsg.write(payload)

    def _write_dist(self, disturbance):
        """Write the disturbance torque (3 values x,y,z in Nm) into torqueMsg"""
        payload = messaging.CmdTorqueBodyMsgPayload()
        payload.torqueRequestBody = list(map(float, disturbance))
        self.cmdMsg.write(payload)


    def step(self, action):
        """Advance the simulation by controlStepTime seconds.
        action: 3 numbers (-> torques in x,y,z) in [-1, 1]; scaled by U_MAX to wheel motor torques [Nm].
        Returns the new state (see get_state)."""
        torques = np.clip(action, -1, 1) * U_MAX    # normalized action -> physical torque
        for _ in range(self.n_physics):     # several small physics steps per control step
            self._write_cmd(torques)
            self.t += simTimeStep
            self.scSim.ConfigureStopTime(macros.sec2nano(self.t))
            self.scSim.ExecuteSimulation()
        return self.get_state()


    def get_state(self):
        """Read the current state from the simulation. Returns a dict:
        t                   : simulated time
        attitude            : attitude of the satellite (quaternion, w >= 0)
        att_err             : attitude error of the satellite (quaternion, w >= 0; [1,0,0,0] means at the target)
        angular_velocity    : angular velocity of the satellite [rad/s]
        wheel_speeds        : speed of the 3 reaction wheels [rad/s]"""
        state = self.satellite.scStateOutMsg.read() # true state of satellite without noise
        att_mrp = rbk.MRPswitch(np.array(state.sigma_BN), 1.0)
        angular_velocity = np.array(state.omega_BN_B)
        att_err_mrp = rbk.MRPswitch(np.array(rbk.subMRP(att_mrp, self.targetAtt)), 1.0) # attitude error
        wheel_speeds = np.array(self.rwStateEffector.rwSpeedOutMsg.read().wheelSpeeds)[:3]
        return dict(t=self.t, attitude=mrp_to_quat(att_mrp),
                    att_err=mrp_to_quat(att_err_mrp), angular_velocity=angular_velocity, wheel_speeds=wheel_speeds)