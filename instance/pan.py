from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
import sys
import random
import math


class CityEscape(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.disableMouse()

        # Fullscreen window
        props = WindowProperties()
        props.setFullscreen(True)
        self.win.requestProperties(props)

        # Movement keys (WASD)
        self.keys = {"w": False, "s": False, "a": False, "d": False}
        for key in self.keys:
            self.accept(key, self.set_key, [key, True])
            self.accept(key + "-up", self.set_key, [key, False])

        # Camera rotation keys (arrow keys)
        self.arrows = {"arrow_left": False, "arrow_right": False, "arrow_up": False, "arrow_down": False}
        for key in self.arrows:
            self.accept(key, self.set_arrow, [key, True])
            self.accept(key + "-up", self.set_arrow, [key, False])

        self.accept("escape", sys.exit)

        # Speeds
        self.player_speed = 50
        self.camera_turn_speed = 60  # degrees per second
        self.enemy_speed = 14

        self.game_over = False
        self.victory = False

        # Camera params
        self.camera_distance = 40
        self.camera_angle_h = 0
        self.camera_angle_p = 60

        # Create city ground (flat)
        self.create_city_ground()

        # Create city buildings (on ground)
        self.create_buildings(num_buildings=150)

        # Player cube
        self.player = loader.loadModel("models/box")
        self.player.setScale(1, 1, 2)  # taller cube for player
        self.player.setColor(1, 1, 1, 1)
        self.player.setPos(0, 0, 1)
        self.player.reparentTo(render)

        # Flag cube somewhere in the city
        self.flag = loader.loadModel("models/box")
        self.flag.setScale(0.7, 0.7, 1.5)
        self.flag.setColor(1, 0, 0, 1)
        fx = random.uniform(-900, 900)
        fy = random.uniform(-900, 900)
        self.flag.setPos(fx, fy, 1)
        self.flag.reparentTo(render)

        # Create many aliens roughly 10 seconds away (~500 units) from player
        self.aliens = []
        num_aliens = 40
        for _ in range(num_aliens):
            alien = loader.loadModel("models/box")
            alien.setScale(1, 1, 2)
            alien.setColor(0, 1, 0, 1)
            angle = random.uniform(0, 2 * math.pi)
            distance = 500 + random.uniform(-100, 100)  # about 500 units away
            x = math.cos(angle) * distance
            y = math.sin(angle) * distance
            alien.setPos(x, y, 1)
            alien.reparentTo(render)
            self.aliens.append(alien)

        # Message text on screen
        self.message = OnscreenText(text="", pos=(0, 0), scale=0.1, fg=(1, 0, 0, 1), mayChange=True)

        # Setup sunny lighting & shadows with "sun"
        self.setup_lighting()

        # Set initial camera position
        self.update_camera_pos()

        # Add update task
        self.taskMgr.add(self.update, "update")

    def set_key(self, key, value):
        self.keys[key] = value

    def set_arrow(self, key, value):
        self.arrows[key] = value

    def create_city_ground(self):
        size = 1000
        cm = CardMaker("cityground")
        cm.setFrame(-size, size, -size, size)
        self.ground = render.attachNewNode(cm.generate())
        self.ground.setP(-90)
        self.ground.setPos(0, 0, 0)
        self.ground.setColor(0.3, 0.9, 0.3, 1)  # dark asphalt color

    def create_buildings(self, num_buildings=150):
        for _ in range(num_buildings):
            bldg = loader.loadModel("models/box")
            sx = random.uniform(5, 15)
            sy = random.uniform(5, 15)
            sz = random.uniform(20, 80)  # tall buildings
            bldg.setScale(sx, sy, sz)
            bldg.setColor(random.uniform(0.2, 0.6), random.uniform(0.2, 0.6), random.uniform(0.2, 0.6), 1)
            x = random.uniform(-950, 950)
            y = random.uniform(-950, 950)
            # Make sure building rests on the ground: z = half building height
            bldg.setPos(x, y, sz / 2)
            bldg.reparentTo(render)

    def setup_lighting(self):
        # Sun directional light
        dlight = DirectionalLight('sun')
        dlight.setColor((1.0, 0.95, 0.8, 1))  # warm sunny color
        dlight.setShadowCaster(True, 2048, 2048)
        dlight.getLens().setFilmSize(100, 100)
        dlight.getLens().setNearFar(10, 2000)
        dlight_np = render.attachNewNode(dlight)
        dlight_np.setHpr(-40, -60, 0)  # sun angle

        render.setLight(dlight_np)

        # Ambient light for soft shadows
        alight = AmbientLight('ambient')
        alight.setColor((0.3, 0.3, 0.3, 8))
        alight_np = render.attachNewNode(alight)
        render.setLight(alight_np)

        render.setShaderAuto()

    def update_camera_pos(self):
        rad_h = math.radians(self.camera_angle_h)
        rad_p = math.radians(self.camera_angle_p)

        x = self.camera_distance * math.cos(rad_p) * math.sin(rad_h)
        y = -self.camera_distance * math.cos(rad_p) * math.cos(rad_h)
        z = self.camera_distance * math.sin(rad_p)

        player_pos = self.player.getPos()
        cam_pos = player_pos + Vec3(x, y, z)
        self.camera.setPos(cam_pos)
        self.camera.lookAt(player_pos + Vec3(0, 0, 2))  # look slightly above player

    def update(self, task):
        if self.game_over or self.victory:
            return Task.cont

        dt = globalClock.getDt()
        speed = self.player_speed * dt
        turn_speed = self.camera_turn_speed * dt

        # Camera rotation with arrow keys
        if self.arrows["arrow_left"]:
            self.camera_angle_h += turn_speed * 2
        if self.arrows["arrow_right"]:
            self.camera_angle_h -= turn_speed * 2
        if self.arrows["arrow_up"]:
            self.camera_angle_p += turn_speed
            if self.camera_angle_p > 85:
                self.camera_angle_p = 85
        if self.arrows["arrow_down"]:
            self.camera_angle_p -= turn_speed
            if self.camera_angle_p < 10:
                self.camera_angle_p = 10

        self.update_camera_pos()

        # Player movement with WASD relative to camera
        move_vec = Vec3(0, 0, 0)
        if self.keys["w"]:
            move_vec += Vec3(0, 1, 0)
        if self.keys["s"]:
            move_vec += Vec3(0, -1, 0)
        if self.keys["a"]:
            move_vec += Vec3(-1, 0, 0)
        if self.keys["d"]:
            move_vec += Vec3(1, 0, 0)

        if move_vec.length() > 0:
            move_vec.normalize()
            rad = math.radians(self.camera_angle_h)
            rotated_x = move_vec.x * math.cos(rad) - move_vec.y * math.sin(rad)
            rotated_y = move_vec.x * math.sin(rad) + move_vec.y * math.cos(rad)
            new_pos = self.player.getPos() + Vec3(rotated_x, rotated_y, 0) * speed

            # Clamp player to city ground bounds
            new_pos.setX(max(min(new_pos.x, 990), -990))
            new_pos.setY(max(min(new_pos.y, 990), -990))
            new_pos.setZ(1)  # keep on ground level
            self.player.setPos(new_pos)

        # Aliens chase player
        for alien in self.aliens:
            direction = self.player.getPos() - alien.getPos()
            direction.setZ(0)
            if direction.length() > 0.5:
                direction.normalize()
                alien.setPos(alien.getPos() + direction * self.enemy_speed * dt)
                # keep alien on ground level
                pos = alien.getPos()
                pos.setZ(1)
                alien.setPos(pos)

            # Check if alien caught player
            if (self.player.getPos() - alien.getPos()).length() < 2.0:
                self.game_over = True
                self.message.setText("👾 Game Over! The aliens got you!")

        # Win condition (reaching flag)
        if (self.player.getPos() - self.flag.getPos()).length() < 3.0:
            self.victory = True
            self.message.setText("🚩 You Win! You found the flag!")

        return Task.cont


if __name__ == "__main__":
    app = CityEscape()
    app.run()

