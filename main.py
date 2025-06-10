from game import Game
from rede import Rede

def main():
    game = Game()
    Rede.clear_results()

    generation = 0
    best_individual = None
    best_score = 0

    while True:
        generation += 1
        print(f"\nIniciando Geração {generation}")

        # Testa 5 indivíduos
        individuals = [Rede.train_neural_network() for _ in range(5)]
        generation_best_score = 0
        generation_best_individual = None

        for i, model in enumerate(individuals):
            score = game.game_loop(model, generation, i + 1)

            if score > generation_best_score:
                generation_best_score = score
                generation_best_individual = model

        print(f"Melhor pontuação da geração {generation}: {generation_best_score}")

        if generation_best_score > best_score:
            best_score = generation_best_score
            best_individual = generation_best_individual

        individuals = Rede.generate_individuals(best_individual)
        Rede.clear_results()
        game.save_individual_info(f"G{generation}_Best", best_score, best_individual.get_weights())

if __name__ == "__main__":
    main()
