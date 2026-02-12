"""
© 2026 Alexandra Mikhael. All Rights Reserved.
"""
import pygame
import numpy as np
import random
import pickle
from collections import deque
import os

import numpy as np

class LeakyIntegrator:
    def __init__(self, alpha=0.95):
        """
        alpha in (0,1): higher = longer memory
        """
        self.alpha = float(alpha)
        self.accumulated_probability = 0.5  # start neutral

    def update(self, new_probability):
        """
        Accepts scalar, list, or ndarray; returns scalar in [0,1].
        """
        # Coerce to scalar
        p = np.asarray(new_probability).astype(float)
        if p.size == 0:
            # nothing to update with; keep previous
            return self.accumulated_probability
        p = float(p.ravel()[0])  # take first element if array/list

        # clip to [0,1] for safety
        if np.isnan(p):
            p = self.accumulated_probability
        p = max(0.0, min(1.0, p))

        self.accumulated_probability = (
            self.alpha * self.accumulated_probability + (1.0 - self.alpha) * p
        )
        return self.accumulated_probability


def display_multiple_mess_udp(messages,colors,offsets,duration=13,logger=None):

	font = pygame.font.SysFont(None,96)
	end_time = pygame.time.get_ticks() + duration*1000
	
	udp_sent=False
	while pygame.time.get_ticks() < end_time:
		pygame.display.get_surface().fill((0,0,0))
		for i, text in enumerate(messages):
			message = font.render(text, True, colors[i])
			pygame.display.get_surface().blit(message,(pygame.display.get_surface().get_width() // 2 - message.get_width() // 2, pygame.display.get_surface().get_height() // 2 - message.get_height() // 2 + offsets[i]))
		
		pygame.display.flip()
		
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				exit()
				
		pygame.time.Clock().tick(60)
		
		
