
def make_annealer(min_epsilon=0.05):
    epsilon = 1.0

    def annealer(step):
        nonlocal epsilon, min_epsilon
        if epsilon > min_epsilon:
            update = 1. / 10000.
            epsilon -= update
        return epsilon

    return annealer


x = np.arange(10000)
annealer = make_annealer()
y = list(map(annealer, x))
plt.scatter(x, y)
plt.show()
