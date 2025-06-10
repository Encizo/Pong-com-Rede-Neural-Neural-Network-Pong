import pygame
import random
import numpy as np

# Configurações básicas do pygame
pygame.init()
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RAQUETE_WIDTH = 25
RAQUETE_HEIGHT = 80
BOLA_SIZE = 25
RAQUETE_SPEED = 25
BOLA_SPEED_X = 15
BOLA_SPEED_Y = 15
RESULTS_FILE = "resultados.txt"


class Bola:
    def __init__(self):
        self.rect = pygame.Rect(SCREEN_WIDTH // 2 - BOLA_SIZE // 2, SCREEN_HEIGHT // 2 - BOLA_SIZE // 2, BOLA_SIZE, BOLA_SIZE)
        self.speed_x = BOLA_SPEED_X * random.choice([1, -1])
        self.speed_y = BOLA_SPEED_Y * random.choice([1, -1])

    def move(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        # Verifica colisões com a parede superior e inferior
        if self.rect.top <= 0 or self.rect.bottom >= SCREEN_HEIGHT:
            self.speed_y *= -1

        # Verifica se a bola atingiu a parede da esquerda
        if self.rect.left <= 0:
            self.speed_x *= -1


    def reset(self):
        self.__init__()

class Raquete:
    def __init__(self, x_position):
        self.rect = pygame.Rect(x_position, SCREEN_HEIGHT // 2 - RAQUETE_HEIGHT // 2, RAQUETE_WIDTH, RAQUETE_HEIGHT)

    def move(self, direction):
        if direction == 'up' and self.rect.top > 0:
            self.rect.y -= RAQUETE_SPEED
        elif direction == 'down' and self.rect.bottom < SCREEN_HEIGHT:
            self.rect.y += RAQUETE_SPEED

    def get_front_hitbox(self):
        return pygame.Rect(self.rect.x, self.rect.y, RAQUETE_WIDTH, self.rect.height)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pong com Rede Neural")

    def draw_screen(self, bola, raquete, score_tela):
        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, WHITE, raquete.rect)
        pygame.draw.ellipse(self.screen, WHITE, bola.rect)

        font = pygame.font.SysFont('Arial', 20)
        text = font.render(f'Pontos: {score_tela}', True, WHITE)
        self.screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, 10))

        pygame.display.flip()

    def get_game_state(self, bola, raquete):
        dist_vertical = bola.rect.centery - raquete.rect.centery
        dist_horizontal = bola.rect.centerx - raquete.rect.centerx
        return np.array([raquete.rect.top, dist_vertical, dist_horizontal, bola.speed_y])

    def save_individual_info(self, name, score, model_weights):
        with open(RESULTS_FILE, 'a') as file:
            file.write(f"{name}: Pontuação = {score}\n")
            file.write(f"Pesos do modelo: {model_weights}\n")

    def game_loop(self, model, generation, individual_number):
        score = 0
        score_tela = 0
        bola = Bola()
        raquete = Raquete(SCREEN_WIDTH - 20)
        clock = pygame.time.Clock()
        game_over = False

        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            bola.move()

            # Obtendo os dados do jogo
            game_state = self.get_game_state(bola, raquete)
            predictions = model.predict(game_state.reshape(1, -1))[0]

            # Definindo o movimento da raquete com base na saída da rede neural
            if predictions[0] > 0.5:
                raquete.move('up')
            elif predictions[1] > 0.5:
                raquete.move('down')

            # Verifica colisão da bola com a raquete
            if bola.rect.colliderect(raquete.get_front_hitbox()):
                score += 4
                score_tela += 1
                bola.speed_x *= -1

            # Recompensa adicional: A raquete está mais alinhada com a bola
            if abs(bola.rect.centery - raquete.rect.centery) < 10: 
                score += 1 

            # Verifica se a bola passou da raquete
            if bola.rect.right >= SCREEN_WIDTH:
                game_over = True

            self.draw_screen(bola, raquete, score_tela)
            clock.tick(60)

        # Salvando estado do individuo no momento em que perde o jogo
        self.save_individual_info(f"G{generation}_I{individual_number}", score, model.get_weights())
        return score