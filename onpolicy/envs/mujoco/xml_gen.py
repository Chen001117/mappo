def get_xml(dog_num=1, obs_num=1):
    strings = \
    """
<mujoco model="navigation">
  
  <size njmax="3000" nconmax="1500"/>
  <option integrator="RK4" timestep="0.01" collision="predefined"/>

  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="5 5 5" type="plane" material="MatPlane"/>
    """
    for i in range(dog_num):
      strings += \
    """
    <body name="dog" pos="0 0 0">
      <site name="dog" pos="0 0 0"/>
      <camera name="camera" mode="trackcom" pos="0 0. 10." xyaxes="1 0 0 0 1 0"/>
      <joint axis="1 0 0" limited="false" name="dog_axisx" pos="0 0 0" type="slide"/>
      <joint axis="0 1 0" limited="false" name="dog_axisy" pos="0 0 0" type="slide"/>
      <joint axis="0 0 1" limited="false" name="dog_rootz" pos="0 0 0" type="hinge"/>
      <joint axis="0 0 1" limited="false" name="dog_axisz" pos="0 0 0" type="slide"/>
      <geom mass="8." size="0.325 0.15 0.15" name="dog" type="box" rgba="0.8 0.4 0. 1" friction="1 0.005 0.001" />
      </body>
    """
    # <geom mass="13" size="0.325 0.15 0.15" name="dog" type="box" rgba="0.8 0.4 0. 1" friction="1 0.005 0.0001" />
    
    
    for i in range(obs_num):
      strings += \
    """
    <body name="obstacle{:02d}" pos="0 0 0">
      <joint axis="1 0 0" limited="false" name="obs{:02d}_axisx" pos="0 0 0" type="slide"/>
      <joint axis="0 1 0" limited="false" name="obs{:02d}_axisy" pos="0 0 0" type="slide"/>
      <joint axis="0 0 1" limited="false" name="obs{:02d}_rootz" pos="0 0 0" type="hinge"/>
      <joint axis="0 0 1" limited="false" name="obs{:02d}_axisz" pos="0 0 0" type="slide"/>
      <geom mass="1000" size="0.5 0.5 0.5" name="obstacle{:02d}" type="box" rgba="0. 0. 1. 1" friction="1 0.005 0.0001" />
    </body>
    """.format(i,i,i,i,i,i)

    strings += \
    """
    <body name="destination" pos="0 0 0">
      <site name="destination" pos="0 0 0"/>
      <joint axis="1 0 0" limited="false" name="destinationx" pos="0 0 0" type="slide"/>
      <joint axis="0 1 0" limited="false" name="destinationy" pos="0 0 0" type="slide"/>
      <geom name="destination" pos="0 0 0" size="0.01 0.01 0.01" type="box"/>
    </body>

  </worldbody>

  <tendon>
    <spatial width="0.03" rgba=".95 .0 .0 1" limited="true" range="0 100."> 
      <site site="dog"/>
      <site site="destination"/>
    </spatial>
  </tendon>

  <actuator>
    <motor ctrllimited="true" ctrlrange="-100.0 100.0" joint="dog_axisx"/>
    <motor ctrllimited="true" ctrlrange="-100.0 100.0" joint="dog_axisy"/>
    <motor ctrllimited="true" ctrlrange="-100.0 100.0" joint="dog_rootz"/>
  </actuator>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".4 .5 .6" rgb2="0 0 0" width="100" height="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="5 5" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>

  <contact>
  <pair geom1="dog" geom2="floor"/>
  """

    for i in range(obs_num):
      strings += \
  """
  <pair geom1="dog" geom2="obstacle{:02d}"/>
  <pair geom1="obstacle{:02d}" geom2="floor"/>
  """.format(i,i)

    strings += \
  """
  </contact>
  </mujoco>
  """
    return strings