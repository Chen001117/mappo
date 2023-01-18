def get_xml(seed):
    strings = \
    """
<mujoco model="navigation">
  
  <size njmax="3000" nconmax="1500"/>
  <option integrator="RK4" timestep="0.01"/>

  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom name='clutter' pos='0.0 0 -0.01' hfield='hf_asset' type='hfield' condim='3' conaffinity='1'/>
    <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="5 5 5" type="plane" material="MatPlane"/>
    <body name="dog" pos="0 0 0.5">
      <site name="dog" pos="0 0 0"/>
      <camera name="camera" mode="trackcom" pos="0 0. 10." xyaxes="1 0 0 0 1 0"/>
      <joint axis="1 0 0" limited="false" name="axisx" pos="0 0 0" type="slide"/>
      <joint axis="0 1 0" limited="false" name="axisy" pos="0 0 0" type="slide"/>
      <joint axis="0 0 1" limited="false" name="rootz" pos="0 0 0" type="hinge"/>
      <joint axis="0 0 1" limited="false" name="axisz" pos="0 0 0" type="slide"/>
      <geom mass="8." size="0.25 0.25 " name="dog" type="cylinder" rgba="0.8 0.4 0. 1" friction="1 0.005 0.0001" />
    </body>

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
    <motor ctrllimited="true" ctrlrange="-100.0 100.0" joint="axisx"/>
    <motor ctrllimited="true" ctrlrange="-100.0 100.0" joint="axisy"/>
    <motor ctrllimited="true" ctrlrange="-100.0 100.0" joint="rootz"/>
  </actuator>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".4 .5 .6" rgb2="0 0 0" width="100" height="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="5 5" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
    <hfield name='hf_asset' size="5 5 2. 0.01" file="{}"/>
  </asset>

</mujoco>
    """.format(seed)
    return strings