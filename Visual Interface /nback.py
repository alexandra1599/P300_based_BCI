"""
© 2026 Alexandra Mikhael. All Rights Reserved.
"""

import pygame
import random
import time
import struct
import threading
import pyautogui
import socket
import UTIL_marker_stream
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream, local_clock

# Mouse Press rigth : yes/Match , left : no/non-match

def send_udp_message(socket, ip, port, message):
    """
    Send a UDP message to the specified IP and port.
    Parameters:
        socket (socket.socket): The socket object for communication.
        ip (str): The target IP address.
        port (int): The target port.
        message (str): The message to send.
    """
    socket.sendto(message.encode('utf-8'), (ip, port))
    print(f"Sent UDP message to {ip}:{port}: {message}")
    
    
# Initialize pygame
pygame.init()

# Screen settings
screen_tmp = pyautogui.size();
screen_width = screen_tmp[0];
screen_height = screen_tmp[1];


screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("N-back Task")

# Set up fonts
font = pygame.font.SysFont('Arial', 100)
letter_font = pygame.font.SysFont('Arial', 150)
small_font = pygame.font.SysFont('Arial', 100)

# Setup UDP
udp_marker = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
message1 = '0' #"Trial Start"
message2 = '100' #"Button press match"
message3 = '200' #"Button press no-match"
message4 = '300' #"Timeout"
message5 =  '400' #"Trial End"
message6 = '1' 
message7 = '2' 
message8 ='11' #match
message9 = '12' #non match
ip = '127.0.0.1'
port = 12345

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

# List of possible numbers
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# N-back level
N = 2 # Change to desired N level


# Create LSL stream for markers
info = StreamInfo('MarkerStream', 'Markers', 2, 0, 'float32', 'marker_stream_id')  # Sending marker and timestamp as a 2-element sample
outlet = StreamOutlet(info)

# Function to display text
def display_text(text, font, color, position, duration=None):
    screen.fill(BLACK)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=position)
    screen.blit(text_surface, text_rect)
    pygame.display.flip()
    if duration:
        time.sleep(duration)

        
# Function to draw fixation cross
def draw_fixation_cross(duration,color):
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < duration:
        screen.fill(BLACK)
        center = (screen_width // 2, screen_height // 2)
        line_width = 5
        pygame.draw.line(screen, color, (center[0] - 20, center[1]), (center[0] + 20, center[1]), line_width)
        pygame.draw.line(screen, color, (center[0], center[1] - 20), (center[0], center[1] + 20), line_width)
        pygame.display.flip()

# Function to generate N-back sequence correctly
def generate_nback_sequence(total_trials, target_ratio, N):
    numTargets = round(target_ratio * total_trials)
    sequence = []
    target_indices = set(random.sample(range(N, total_trials), numTargets))  # Pick where matches should be

    for i in range(total_trials):
        if i in target_indices: 
            number = sequence[i - N]  # Match N-back number
        else:
            # Ensure no accidental match
            number = random.choice([l for l in numbers if i < N or l != sequence[i - N]])

        sequence.append(number)

    print("\nGenerated Sequence:", sequence)
    print("Target Indices:", target_indices)  # Debug: See where matches occur
    return sequence, target_indices

# Modify the trial section to ensure consistent length for each trial.
def run_nback_task():
    total_trials = 50
    target_ratio = 0.3  # 30% of trials should be matches
    sequence, target_indices = generate_nback_sequence(total_trials, target_ratio, N)
    correct_responses = 0
    trial = 0
    
    # Display "Press any key to start"
    display_text("Press any key to start", font, WHITE, (screen_width // 2, screen_height // 2), 0)
    waiting_for_start = True
    while waiting_for_start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                waiting_for_start = False

    while trial < total_trials:
        number = sequence[trial]

        if trial > 0:  # Ensure at least one previous trial exists
            correct_number = sequence[trial - 1] if N == 0 else sequence[trial - N]
            is_match = trial in target_indices  # Check if this trial is a match
            
            if (is_match):
            	send_udp_message(udp_marker, ip, port, message8)
            elif(not is_match):
            	send_udp_message(udp_marker, ip, port, message9)
                    
            start_time = time.time()
            response_time = start_time + 1.1
            response = None
            send_udp_message(udp_marker, ip, port, message1)

            while time.time() - start_time < 1.1:
            
            	#Only show the number for 500 ms
            	if time.time() - start_time < 0.5 :
            		screen.fill(BLACK)
            		text_surface = letter_font.render(number, True, WHITE)
            		text_rect = text_surface.get_rect(center=(screen_width // 2, screen_height // 2))
            		screen.blit(text_surface, text_rect)
            		pygame.display.flip()
            	else:
            		# Clear screen after 600ms for duration
            		duration = 0.8
            		draw_fixation_cross(duration,WHITE)
            		pygame.display.flip()
            		

            		
            	for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                    if event.type == pygame.MOUSEBUTTONDOWN and response is None and time.time() - start_time > 0.3:
                        
                        if event.button in (3, 2):
                            response = 'y'  # Yes (match)
                            send_udp_message(udp_marker, ip, port, message2)
                        elif  event.button == 1:
                            response = 'n'  # No (no match)
                            send_udp_message(udp_marker, ip, port, message3)

            if response is None:
                response = 'timeout'
                send_udp_message(udp_marker, ip, port, message4)

            if response != 'timeout':
                if (response == 'y' and is_match) or (response == 'n' and not is_match):
                    correct_responses += 1
                    send_udp_message(udp_marker, ip, port, message6)
                else:
                    send_udp_message(udp_marker, ip, port, message7)
         

        trial += 1
        send_udp_message(udp_marker, ip, port, message5)
        draw_fixation_cross(1.5,GREEN)
        
        draw_fixation_cross(0.8,WHITE)

    # Show final score
    display_text(f'You got {correct_responses} out of {total_trials} correct!', font, WHITE, (screen_width // 2, screen_height // 2), 3)
    pygame.quit()

# Run the N-back task
if __name__ == "__main__":
    run_nback_task()
