'''
Usage:
    python test_games.py -v
    python test_games.py
'''
import unittest

from games.gridworld import Gridworld as Game


class TestGames(unittest.TestCase):

    def setUp(self):
        self.game = Game(size=4, mode='static')

    def test_static_game_win(self):
        for m in ['d'] * 2 + ['l'] * 3 + ['u'] * 2:
            self.assertEqual(-1, self.game.reward())
            self.game.makeMove(m)
        self.assertEqual(10, self.game.reward())

    def test_static_game_loss(self):
        for m in ['l'] * 2:
            self.assertEqual(-1, self.game.reward())
            self.game.makeMove(m)
        self.assertEqual(-10, self.game.reward())


if __name__ == '__main__':
    unittest.main()
