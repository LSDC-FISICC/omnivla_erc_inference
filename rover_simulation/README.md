# rover_simulation

Herramientas para probar el control y la evasión de obstáculos **antes de ir al campo**, y para
revisar los bags **después**. Aquí salieron los números de `docs/MissionCarrotSidestep.md` y
`docs/MissionWuhanHard.md` (repo `Earth-rover-ros2-bridge`).

## Dónde está cada cosa

| | |
|---|---|
| **Motor del simulador** | `../erc_inference/test/controller_sim.py` (rover, localización, bucle de misión) y `../erc_inference/test/obstacles.py` (mundos y perfil de percepción simulado). Se quedan en `test/` porque `test_controller_sim.py` y `colcon test` los importan de ahí. |
| **Código del robot que se ejecuta** | `../erc_inference/erc_inference/`: `motion_control.py`, `sidestep.py`, `local_planner.py` y la geometría de ruta de `checkpoint_controller_node.py`. Es el mismo código del rover, no una copia. |
| `experiment_simple.py` | **Empieza aquí.** Un experimento en imágenes: costmap global (OSM) + ruta A\*, espacio libre, costmap local al replanificar y al final, trayectoria. |
| `common.py` | Rutas y `run()`: una misión simulada en una llamada. |
| `sweep.py` | Barrido configuraciones × escenarios × semillas, en paralelo. |
| `plot_run.py` | Una misión a PNG: el mapa local construido, obstáculos reales, trayectoria y replanificaciones. |
| `turn_check.py` | Cuánto gira de verdad la maniobra de esquiva en simulación. |
| `e2e/` | Prueba de punta a punta **en ROS** con los nodos reales, un rover falso y un SDK falso. |
| `bags/` | Análisis de bags de campo: mapa local desde el bag, vista tipo RViz, dinámica de giro, imagen de cada freno. |

## Qué modela el simulador, y de dónde sale cada número

- **Planta:** retardo de 1.3 s; k_ω 1.18 en el sitio y 0.36 en marcha; velocidades mínimas para
  arrancar (Tarea B, `mission_16sept`). Reproduce el sobregiro del 22-sept: 103° contra ~100° en
  campo.
- **Localización:** error AR(1) de 0.2 m, sesgo de rumbo ±6°, latencia de sensores.
- **Percepción (`obstacles.profile`):** perfil polar de ±60° y 3 m, periferia recortada a ~1.7 m,
  ruido de 8 cm, 3 % de fallos.
- **Mundos:** `straight`, `L-turn`, `U-turn`, `zigzag` y `mission_16sept`, sin obstáculos; `kerb`,
  `post`, `chicane`, `hedge` (el seto del 22-sept), `planter` y `wall`, con obstáculos.

**Límites.** Las paredes son segmentos finos: el rover puede atravesarlas después de "chocar", y el
choque se cuenta aparte (`hit`). El perfil no tiene los errores de escala de DA3. Todo resultado de
aquí se confirma en campo.

## Uso

Simulación, con el Python del venv (tiene `utm`) y un workspace cargado (mensajes de
`erc_inference_msgs` / `erc_static_map_msgs`):

```bash
source /opt/ros/jazzy/setup.bash && source ~/lsdc_ws/install/setup.bash
cd ~/lsdc/erc-omni-vla/omnivla_erc_inference/rover_simulation
PY=~/lsdc/erc-omni-vla/.venv/bin/python3

$PY experiment_simple.py                                          # -> experiment_simple.png
$PY sweep.py --carrot 1.5 --seeds 8 --configs ss,lp+ss           # la tabla de MissionCarrotSidestep
$PY sweep.py --scenarios wall --seeds 8 --ss '{"stop_distance_m": 1.0}' --fails
$PY plot_run.py planter 0 --out /tmp/planter.png
$PY turn_check.py --ss '{"lead_s": 2.8}'
```

Configuraciones (`--configs`): `carrot` (sin reacción a obstáculos), `ss` (maniobra de esquiva),
`lp` (mapa local + replanificación), `lp+ss` (las dos; lo que se usaría en campo). `--ss` y `--lp`
cambian parámetros de `sidestep.DEFAULTS` y `local_planner.DEFAULTS`.

De punta a punta en ROS (nodos reales desde este árbol de código; logs en `e2e/out/`):

```bash
./e2e/run_e2e.sh hedge 300 true        # local_replan on
./e2e/run_e2e.sh hedge 300 false       # para comparar
```

Bags de campo, con el Python del sistema (tiene `rosbag2_py`):

```bash
source /opt/ros/jazzy/setup.bash
python3 bags/bag_local_map.py ~/lsdc/erc-omni-vla/rosbags/<bag> mapa.png
python3 bags/bag_turns.py ~/lsdc/erc-omni-vla/rosbags/<bag>
python3 bags/bag_brakes.py ~/lsdc/erc-omni-vla/rosbags/<bag> frenos.jpg
```

## Ver una misión en RViz

`checkpoint_controller_node` publica, durante cada tramo, todo en el marco `leg_local` (metros
este/norte desde el inicio del tramo, el mismo marco del carrot):

| tópico | qué es |
|---|---|
| `/erc/global_route` | la ruta A\* del tramo |
| `/erc/local_route` | la ruta actual, con los desvíos (solo con `local_replan`) |
| `/erc/local_costmap` | el mapa local (solo con `local_replan`) |
| `/erc/carrot` | el carrot |
| `/erc/free_space_leg` | `/erc/free_space` en el marco `rover_leg` |
| TF `leg_local → rover_leg` | la pose que usan el carrot y el planificador (GPS filtrado + compass) |

```bash
# en vivo
rviz2 -d $(ros2 pkg prefix erc_inference)/share/erc_inference/rviz/local_planning.rviz
# desde un bag
ros2 bag play <bag> --clock
rviz2 -d $(ros2 pkg prefix erc_inference)/share/erc_inference/rviz/local_planning.rviz --ros-args -p use_sim_time:=true
```

Sin monitor en el DGX (acceso remoto):
- **Foxglove, sin ROS en la laptop.** Los bags son MCAP. `bags/extract_viz.sh <bag>` saca solo los
  tópicos de visualización (sin imágenes: MB en vez de GB); se copia a la laptop y se abre en
  Foxglove. En el panel 3D, fixed frame `leg_local`.
- **Foxglove en vivo:** `sudo apt install ros-jazzy-foxglove-bridge`; en el DGX
  `ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8766`; en la laptop
  `ssh -L 8766:localhost:8766 <dgx>` y en Foxglove abrir `ws://localhost:8766`. Puerto 8766
  porque `e2e/fake_world.py` usa el 8765.
- **RViz en la laptop:** copiar el bag (o el extracto) y `rviz/local_planning.rviz`; solo usa
  mensajes estándar, no hace falta este código.
- **Sin nada:** `bags/bag_leg_view.py <bag> out.png` dibuja lo mismo a PNG (último tramo).

Los bags grabados antes de este cambio no tienen esos tópicos. Para ellos, `bags/bag_local_map.py`
reconstruye el mapa y las replanificaciones a PNG.
