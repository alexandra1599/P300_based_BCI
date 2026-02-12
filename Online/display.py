"""
© 2026 Alexandra Mikhael. All Rights Reserved.
"""
import pygame
import pyautogui
import config
import time
import sys

screen_tmp = pyautogui.size()
screen_width = screen_tmp[0]
screen_height = screen_tmp[1]

screen = pygame.display.set_mode((screen_width, screen_height))


# Function to display text
def display_text(text, font, color, position, duration=None):
    screen.fill(config.BLACK)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=position)
    screen.blit(text_surface, text_rect)
    pygame.display.flip()
    if duration:
        time.sleep(duration)


def draw_cross(surface, color, center=None, size=20, line_width=5):
    # draw the cross ONCE (no loop, no flip)
    if center is None:
        w, h = surface.get_size()
        center = (w // 2, h // 2)
    pygame.draw.line(
        surface,
        color,
        (center[0] - size, center[1]),
        (center[0] + size, center[1]),
        line_width,
    )
    pygame.draw.line(
        surface,
        color,
        (center[0], center[1] - size),
        (center[0], center[1] + size),
        line_width,
    )


def display_cross_with_messages(
    messages, colors, offsets, duration, cross_color, font_obj=None
):
    font_obj = font_obj or pygame.font.SysFont(None, 72)
    start = time.time()
    clock = pygame.time.Clock()
    while time.time() - start < duration:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        screen.fill(config.BLACK)

        # draw cross
        draw_cross(screen, cross_color, size=20, line_width=5)

        # draw messages on top
        for i, text in enumerate(messages):
            surf = font_obj.render(text, True, colors[i])
            rect = surf.get_rect(
                center=(screen_width // 2, screen_height // 2 + offsets[i])
            )
            screen.blit(surf, rect)

        pygame.display.flip()
        clock.tick(60)
