import pygame

from communicate.channel import ChannelFactoryInitialize, ChannelPublisher
from joystick import INITIAL_TARGET_MM, joystick_data


STEP_MM = 10.0
UPDATE_HZ = 10
# Pairs of keys for the positive and negative directions of right XYZ, then left XYZ.
AXIS_KEYS = (
    (pygame.K_w, pygame.K_s),
    (pygame.K_a, pygame.K_d),
    (pygame.K_q, pygame.K_e),
    (pygame.K_i, pygame.K_k),
    (pygame.K_j, pygame.K_l),
    (pygame.K_u, pygame.K_o),
)


def draw_controls(screen, font, target, focused):
    screen.fill((26, 29, 34))
    lines = (
        "Right arm: W/S (X)   A/D (Y)   Q/E (Z)",
        "Left arm: I/K (X)   J/L (Y)   U/O (Z)",
        "1/2: gripper   Space: reset targets   Esc: quit",
        "Right: [{:.0f}, {:.0f}, {:.0f}] mm    Left: [{:.0f}, {:.0f}, {:.0f}] mm".format(
            *target
        ),
        "Click this window to control the robot." if not focused else "Hold a key to move.",
    )
    for row, line in enumerate(lines):
        screen.blit(font.render(line, True, (235, 238, 242)), (16, 14 + row * 27))
    pygame.display.flip()


def main():
    publisher = None
    try:
        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((530, 158))
        pygame.display.set_caption("iRobot Keyboard Teleoperation")
        font = pygame.font.Font(None, 25)
        clock = pygame.time.Clock()

        ChannelFactoryInitialize()
        publisher = ChannelPublisher("topic", joystick_data)
        publisher.Init()
        target = list(map(float, INITIAL_TARGET_MM))

        while True:
            pressed = set()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    pressed.add(event.key)
            if pygame.K_ESCAPE in pressed:
                return

            focused = pygame.key.get_focused()
            gripper_1 = gripper_2 = False
            if focused:
                held = pygame.key.get_pressed()
                if pygame.K_SPACE in pressed:
                    target[:] = INITIAL_TARGET_MM
                else:
                    for axis, (positive, negative) in enumerate(AXIS_KEYS):
                        positive_pressed = held[positive] or positive in pressed
                        negative_pressed = held[negative] or negative in pressed
                        target[axis] += STEP_MM * (positive_pressed - negative_pressed)
                gripper_1 = held[pygame.K_1] or pygame.K_1 in pressed
                gripper_2 = held[pygame.K_2] or pygame.K_2 in pressed

            message = joystick_data(
                matrix_data=[*map(float, target), float(gripper_1 and not gripper_2), float(gripper_2)]
            )
            publisher.Write(message)
            draw_controls(screen, font, target, focused)
            clock.tick(UPDATE_HZ)
    except KeyboardInterrupt:
        pass
    finally:
        if publisher is not None:
            publisher.Close()
        pygame.quit()


if __name__ == "__main__":
    main()
