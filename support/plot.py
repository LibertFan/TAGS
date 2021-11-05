import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
np.random.seed(19680801)


def main():
    data_dir = "/Users/fanzhihao/Documents/Research/NIPS2021"
    kwargs = dict(alpha=0.2, bins=2000, density=True)

    max_d = 4000.0
    data0 = [math.exp(x) for x in json.load(open(os.path.join(data_dir, 'dataset_{}.json'.format(0)), 'r')) if math.exp(x) < max_d]
    print('data0', len(data0))
    plt.hist(data0, color='g', **kwargs)
    data1 = [math.exp(x) for x in json.load(open(os.path.join(data_dir, 'dataset_{}.json'.format(1)), 'r')) if math.exp(x) < max_d]
    print('data1', len(data1))
    plt.hist(data1, color='b', **kwargs)
    data2 = [math.exp(x) for x in json.load(open(os.path.join(data_dir, 'dataset_{}.json'.format(2)), 'r')) if math.exp(x) < max_d]
    print('data2', len(data2))
    plt.hist(data2, color='r', **kwargs)
    # plt.gca().set(title='Frequency Histogram of Diamond Depths', ylabel='Frequency')
    plt.xlim(0, 250)
    plt.show()


def main2():
    data_dir = "/Users/fanzhihao/Documents/Research/NIPS2021"
    import seaborn as sns
    # white, dark, whitegrid, darkgrid, ticks

    sns.set_style("darkgrid")
    # sns.set_style("ticks")

    # Import data
    # df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/diamonds.csv')
    # x1 = df.loc[df.cut == 'Ideal', 'depth']
    # x2 = df.loc[df.cut == 'Fair', 'depth']
    # x3 = df.loc[df.cut == 'Good', 'depth']

    # Plot
    # kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 2})
    #
    # plt.figure(figsize=(10, 7), dpi=80)
    # sns.distplot(x1, color="dodgerblue", label="Compact", **kwargs)
    # sns.distplot(x2, color="orange", label="SUV", **kwargs)
    # sns.distplot(x3, color="deeppink", label="minivan", **kwargs)
    # plt.xlim(50, 75)
    # plt.legend();

    kwargs = dict(alpha=0.8, bins=2000)

    # plt.rcParams['figure.figsize'] = (40.0, 8.0)
    max_d = 1500.0

    kwargs = dict(hist_kws={'alpha': .2}, kde_kws={'linewidth': 2, 'shade': True}, bins=5000)

    plt.figure(figsize=(9.2, 6), dpi=80)
    # data0 = [math.exp(x) for x in json.load(open(os.path.join(data_dir, 'dataset_{}.json'.format(0)), 'r')) if math.exp(x) < max_d]
    data0 = [x for x in json.load(open(os.path.join(data_dir, 'dataset_{}.json'.format(0)), 'r')) if x < max_d]
    print('data0', len(data0))
    line0 = sns.distplot(data0, **kwargs, color='g', hist=False, kde=True, label="Positive Text")
    # data1 = [math.exp(x)+20 for x in json.load(open(os.path.join(data_dir, 'dataset_{}.json'.format(1)), 'r')) if math.exp(x) < max_d]
    data1 = [x for x in json.load(open(os.path.join(data_dir, 'dataset_{}.json'.format(1)), 'r')) if x < max_d]
    print('data1', len(data1))
    line1 = sns.distplot(data1, **kwargs, color='b', hist=False, kde=True, label="Synthetic Text")
    # data2 = [math.exp(x)+10 for x in json.load(open(os.path.join(data_dir, 'dataset_{}.json'.format(2)), 'r')) if math.exp(x) < max_d]
    data2 = [x for x in json.load(open(os.path.join(data_dir, 'dataset_{}.json'.format(2)), 'r')) if x < max_d]
    line2 = sns.distplot(data2, **kwargs, color='r', hist=False, kde=True, label="Corrected Text")
    plt.xlim(0, 300)

    plt.legend(loc="upper right", fontsize=20)

    # plt.xticks(np.arange(-5, 5, 0.5), fontproperties='Times New Roman', size=10)
    # plt.yticks(np.arange(-2, 2, 0.3), fontproperties='Times New Roman', size=10)

    plt.xlabel('', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.ylabel('', fontdict={'family': 'Times New Roman', 'size': 20})

    plt.yticks(np.arange(0.0, 0.012, 0.002), fontproperties='Times New Roman', size=20)
    plt.xticks(np.arange(0, 300, 50), fontproperties='Times New Roman', size=20)
    plt.legend(prop={'family': 'Times New Roman', 'size': 20})
    # plt.rcParams['xtick.direction'] = 'out'
    # plt.rcParams['ytick.direction'] = 'out'
    # plt.bar([0, 50, 100, 150, 200, 250], [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])

    # plt.minorticks_on()
    # plt.tick_params(which='major', width=4, length=10, direction="inout")

    plt.show()

    # plt.gca().set(title='Frequency Histogram of Diamond Depths', ylabel='Frequency')
    # plt.show()
    # plt.legend()


def main3():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde

    data_dir = "/Users/fanzhihao/Documents/Research/NIPS2021"
    max_d = 4000.0
    data0 = [math.exp(x) for x in json.load(open(os.path.join(data_dir, 'dataset_{}.json'.format(0)), 'r')) if math.exp(x) < max_d]

    density = gaussian_kde(data0)
    xs = np.linspace(0, 8, 200)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    plt.plot(xs, density(xs))
    plt.show()


if __name__ == '__main__':
    main2()
