"""Accept X matrix and y vector in command line arguments and perform machine
learning on them using an algorithm of chouce
@uthor: Himaghna, 14th February 2020

"""
from argparse import ArgumentParser
import copy
import datetime
import os.path
import pickle
from pprint import pprint

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.stats import uniform
from sklearn.linear_model import Lasso, LassoCV, HuberRegressor
from sklearn.linear_model import LinearRegression, ElasticNetCV
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import yaml

from helper_files import plot_parity, pretty_plot, plot_density, plot_bivariate


class Model:

    def __init__(
            self, target_label, X_train, y_train, normalize_X, normalize_y,
            seeds):
        self.model_ = None
        self.mode_ = None
        self.task_ = None
        self.sub_models = None
        self.target_label = target_label
        self.seeds = seeds
        if normalize_X:
            print('Normalizing X')
            x_scaler = StandardScaler()
            self.X_train = x_scaler.fit_transform(X_train)
            self.X_scaler = x_scaler
        else:
            self.X_train = X_train
            self.X_scaler = None
        y_train = y_train.reshape(-1, 1)
        if normalize_y:
            print('Normalizing y')
            y_scaler = StandardScaler()
            self.y_train = y_scaler.fit_transform(y_train)
            self.y_scaler = y_scaler
        else:
            self.y_train = y_train
            self.y_scaler = None

    def train_(self, training_algorithm, **train_params):
        self.mode_ = 'train'
        if training_algorithm == 'lasso':
            self.task_ = 'regression'
            self.do_lasso( **train_params)
        elif training_algorithm =='huber':
            self.task_ = 'regression'
            self.do_huber(**train_params)
        elif training_algorithm == 'ordinary_least_square':
            self.task = 'regression'
            self.do_ordinary_least_square(**train_params)
        elif training_algorithm == 'elastic_net':
            self.task = 'regression'
            self.do_elastic_net(**train_params)
        else:
            raise NotImplemented(f'{training_algo} is not implemented!!')
        self.evaluate_model(self.X_train, self.y_train)

    def test_(self, X_test, y_test, **test_params):
        y_test = y_test.reshape(-1, 1)
        if self.model_ is None:
            raise UserWarning('Model not trained!')
        if self.X_scaler is not None:
            X_test = self.X_scaler.transform(X_test)
        if self.y_scaler is not None:
            y_test = self.y_scaler.transform(y_test)
        self.mode_ = 'test'
        self.evaluate_model(
            X_test, y_test, plot_color=test_params.get('plot_color', 'red'))

    def get_active_coeff(self):
        """Get the number of active coefficients in the model"""
        num_active_coeff = 0
        for coefficient in self.model_.coef_:
            if abs(coefficient) > 0:
                num_active_coeff += 1
        return num_active_coeff
               
    def evaluate_model(self, X, y, plot_color=None):
        y_pred = self.model_.predict(X)
        if self.y_scaler is not None:
            y_pred = y_pred * self.y_scaler.scale_ + self.y_scaler.mean_
            y = y * self.y_scaler.scale_ + self.y_scaler.mean_
        if self.mode_ == 'train':
            print('*******Training Fit Report******')
            print(
                ' Training Mean Absolute Error: ' \
                f'{metrics.mean_absolute_error(y_true=y, y_pred=y_pred)}')
            print(
                f'R-sq {metrics.r2_score(y_true=y, y_pred=y_pred)}')
        elif self.mode_ == 'test':
            if plot_color is None:
                plot_color = 'black'
            test_mae = metrics.mean_absolute_error(
                y_true=y.ravel(),y_pred=y_pred.ravel())
            test_r2 = metrics.r2_score(y_true=y, y_pred=y_pred)
            print('*******Testing Fit Report******')
            print(
                'Testing Mean Absolute Error: ' \
                f'{test_mae}')
            print(
                f'R-sq {test_r2}')
            axes = plot_parity(
                    y, y_pred, 
                    xlabel=f'True {self.target_label} (kcal/mol)',
                    xlabel_fontsize=16,
                    ylabel=f'{self.target_label} (kcal/mol)',
                    ylabel_fontsize=16,
                    c=plot_color, alpha=0.4, s=100, offset=1,
                    show_plot=False)
            axes.set_aspect('equal', adjustable='box')
            
            # Add metrics to plot
            plt.text(
                0.05, 0.9, s=f'Subgraphs: {self.get_active_coeff()}',
                transform=axes.transAxes, fontsize=16)
                    
            plt.text(
                0.05, 0.8, s=f'R\u00b2: {test_r2 : .2f}',
                transform=axes.transAxes, fontsize=16)
            
            plt.text(
                0.05, 0.7, s=f'MAE: {test_mae : .2f} kcal/mol',
                transform=axes.transAxes, fontsize=16)
            plt.text(
                0.05, 0.6, s=f'{X.shape[0]} Molecules',
                transform=axes.transAxes, fontsize=16)
            plt.tight_layout()
            plt.show()

            # density plots
            plot_density(y.ravel(),
                color=plot_color, label='Ground Truth Distribution')
            plot_density(
                y_pred, color='#fb0091', label='Predicted Distribution',
                shade=False, xlabel=self.target_label+' (kcal/mol)')
            plt.tight_layout()
            plt.show()
            plot_bivariate(
                y.ravel(), y_pred, xlabel='Ground Truth Distribution',
                y_label='Predicted Distribution')
            plt.tight_layout()
            plt.show()


    def set_sub_models(self):
        """Train an array of models trained on subset of data. This is used
        to calculate learning curves. The sizof the sub-data grid is
        [10%, 20% ... 90% of original data size].
        This assumes that self.model_ object is created but not trained.
        It then makes multiple deepcopies of sel.model_and thus retains the
        same training parameters


        """
        sub_data_grid = [0.1 * i for i in range(1, 10)]
        self.sub_models = [
        copy.deepcopy(self.model_) for _ in range(len(sub_data_grid))]
        # fit sub-models to subset of data
        for key, data_size in enumerate(sub_data_grid):
            X, _, y, _ = train_test_split(
                self.X_train, self.y_train.ravel(),
                train_size=data_size,
                random_state=self.seeds.get('l_curve_seed', 22))
            self.sub_models[key].fit(X, y)

    def plot_learning_curve(self, X_test, y_test, plot_color):
        """Plot a learning curve by measuring performance on a test set
        by taking increasing ratios of the training data

        Parameters
        ---------
        X_test : (n x p) numpy array
            Data matrix of testing data.
        y_test : (n x 1) numpy array
            Vector of responses for training set.
        plot_color : str
            Color used to plot.

        """
        if self.X_scaler is not None:
            X_test = self.X_scaler.transform(X_test)

        maes = []
        for model_ in self.sub_models:
            scale_ = self.y_scaler.scale_ if self.y_scaler is not None else 1
            mean_ = self.y_scaler.mean_ if self.y_scaler is not None else 0
            y_pred = (model_.predict(X_test) * scale_ + mean_)
            maes.append(metrics.mean_absolute_error(y_true=y_test,
                                                    y_pred=y_pred))
        maes.append(metrics.mean_absolute_error(
                        y_true=y_test, 
                        y_pred=(self.model_.predict(X_test) * scale_ + mean_)))
        pretty_plot(
            x=[0.1 * i for i in range(1, 11)], y=maes,
            xlabel='Training Data Fraction',
            ylabel=self.target_label+' MAE (kcal/mol)',
            marker='s', markerfacecolor=plot_color,
            markeredgecolor='black', c=plot_color, markersize=30,
            markeredgewidth=2, xticksize=24, yticksize=24)
              # alternate axes
        ax = plt.gca()
        secax = ax.secondary_xaxis('top', functions=(
                                         lambda x: x * self.X_train.shape[0],
                                         lambda x: x / self.X_train.shape[0]))
        secax.set_xlabel('Number of Training Molecules', fontsize=20)
        secax.set_xticklabels([0.1 * i * self.X_train.shape[0] 
                                  for i in range(1, 11)],
                              fontsize=20)
        plt.show()

    # algorithms available
    def do_lasso(self, **cross_valid_params):
        print('Regressing using LASSO')
        default_cross_valid_params = {
            'max_iter': 20000,
            'random_state': 1,
            'cv': 5}
        default_cross_valid_params.update(cross_valid_params)
        print('***CROSS VALIDATION PARAMS')
        pprint(default_cross_valid_params)
        self.model_ = LassoCV(
            cv=default_cross_valid_params['cv'],
            random_state=default_cross_valid_params['random_state'],
            max_iter=default_cross_valid_params['max_iter'])
        if cross_valid_params.get('get_learning_curve', False):
            # create submodels trained on subsets of data for learning curve
            self.set_sub_models()
        self.model_.fit(X=self.X_train, y=self.y_train.ravel())

    def do_huber(self, **cross_valid_params):
        print('Regressing using HUBER')
        default_cross_valid_params = {
            'max_iter': 20000,
            'cross_valid_iters': 2,
            'random_state': 1,
            'cv': 2}
        default_cross_valid_params.update(cross_valid_params)
        cross_valid_param_grid = {
            'epsilon' : uniform(loc=1, scale=20),
            'alpha': [10 ** power_ for power_ in uniform.rvs(
                                              loc=-6., scale=6.,
                                              size=default_cross_valid_params[
                                                'cross_valid_iters'])]}
        self.model_ = HuberRegressor()
        if cross_valid_params.get('get_learning_curve', False):
            # create submodels trained on subsets of data for learning curve
            self.set_sub_models()
        self.model_.fit(X=self.X_train, y=self.y_train.ravel())
    
    def do_ordinary_least_square(self, **cross_valid_params):
        print('Regressing using Ordinary Least Squares')
        self.model_ = LinearRegression()
        if cross_valid_params.get('get_learning_curve', False):
            # create submodels trained on subsets of data for learning curve
            self.set_sub_models()
        self.model_.fit(X=self.X_train, y=self.y_train.ravel())
    
    def do_elastic_net(self, **cross_valid_params):
        print('Regressing using Elastic Net')
        default_cross_valid_params = {
            'max_iter': 20000,
            'random_state': 1,
            'cv': 5}
        default_cross_valid_params.update(cross_valid_params)
        print('***CROSS VALIDATION PARAMS')
        pprint(default_cross_valid_params)
        self.model_ = ElasticNetCV(
            cv=default_cross_valid_params['cv'],
            random_state=default_cross_valid_params['random_state'],
            max_iter=default_cross_valid_params['max_iter'])
        if cross_valid_params.get('get_learning_curve', False):
            # create submodels trained on subsets of data for learning curve
            self.set_sub_models()
        self.model_.fit(X=self.X_train, y=self.y_train.ravel())
        self.evaluate_model(self.X_train, self.y_train)
     


if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('config', help='Configuration yaml file')
    parser.add_argument(
        '-m', '--mode', default='train', required=False, help='[test, train]')
    args = parser.parse_args()
    config_path = args.config
    mode = args.mode
    # load stuff
    configs = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    testing_data = configs['testing_data']
    plot_color = testing_data.get('plot_color', None)
    
    if mode == 'train':
        training_data = configs['training_data']
        training_parameters = configs['training_parameters']
        if testing_data.get('split_from_training', None):
            # create test from train data
            split_config = testing_data['split_from_training']
            X = pickle.load(open(training_data['X'], "rb"))
            y = pickle.load(open(training_data['y'], "rb"))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=split_config['test_size'],
                random_state=split_config['random_state'])
        else:
            X_train = pickle.load(open(training_data['X'], "rb"))
            y_train = pickle.load(open(training_data['y'], "rb"))
            X_test = pickle.load(open(testing_data['X'], "rb"))
            y_test = pickle.load(open(testing_data['y'], "rb"))
        model = Model(
            target_label='\u0394'+training_data.get('target_label', 'Response'),
            X_train=np.array(X_train),
            normalize_X=training_data.get('normalize_X'),
            y_train=y_train, 
            normalize_y=training_data.get('normalize_y'), 
            seeds=configs.get('random_seeds', None))
        model.train_(
            training_algorithm=training_parameters['algorithm'] ,
            cv=training_parameters.get('cross_validation_folds', None),
            get_learning_curve=training_parameters.get(
                'get_learning_curve', False),
            cross_valid_iters=training_parameters.get('cross_valid_iters', 2))
        model.test_(np.array(X_test), y_test, plot_color=plot_color)
        model.plot_learning_curve(X_test, y_test, plot_color=plot_color)
        if training_parameters.get('store_model', False):
            d = datetime.datetime.today()
            model_path = os.path.join(
                os.path.dirname(training_data['X']),
                f"model_{training_parameters['algorithm']}_" \
                  f"{training_data.get('target_label', 'Response')}_" \
                      f"{d.month}_{d.day}.p")
            print(f'Storing model at {model_path}')
            pickle.dump(model, open(model_path, "wb"))
    else:
        # testing
        print('Operating in Testing Mode')
        model_path = testing_data.get('model', None)
        if model_path is None:
            print('No model path specified!')
            exit()
        model = pickle.load(open(model_path, "rb"))
        X_test = pickle.load(open(testing_data['X'], "rb"))
        y_test = pickle.load(open(testing_data['y'], "rb"))
        model.test_(X_test, y_test, plot_color=plot_color)
        model.plot_learning_curve(X_test, y_test, plot_color=plot_color)













