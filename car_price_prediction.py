import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class CarPricePredictionModel:
    """
    A machine learning model to predict car prices based on features
    """
    
    def __init__(self, filepath):
        """Initialize the model with a dataset"""
        self.df = pd.read_csv(filepath)
        self.models = {}
        self.predictions = {}
        self.scaler = StandardScaler()
        
    def display_dataset_info(self):
        """Display information about the dataset"""
        print("=" * 70)
        print("CAR PRICE PREDICTION - DATASET OVERVIEW")
        print("=" * 70)
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"\nColumn Names:\n{self.df.columns.tolist()}")
        print(f"\nFirst Few Rows:\n{self.df.head()}")
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        print(f"\nStatistical Summary:\n{self.df.describe()}")
        
    def analyze_features(self):
        """Analyze the relationship between features and price"""
        print("\n" + "=" * 70)
        print("FEATURE ANALYSIS")
        print("=" * 70)
        
        # Calculate correlation with price
        if 'Price' in self.df.columns:
            correlation = self.df.corr()['Price'].sort_values(ascending=False)
            print(f"\nCorrelation with Price:\n{correlation}")
            
            # Identify strong correlations
            strong_correlations = correlation[abs(correlation) > 0.5]
            print(f"\nStrong Correlations (|r| > 0.5):\n{strong_correlations}")
            
    def preprocess_data(self):
        """Preprocess data for model training"""
        print("\n" + "=" * 70)
        print("DATA PREPROCESSING")
        print("=" * 70)
        
        # Handle missing values
        self.df = self.df.fillna(self.df.mean(numeric_only=True))
        print(f"\n✓ Missing values handled")
        
        # Separate features and target
        if 'Price' in self.df.columns:
            self.X = self.df.drop('Price', axis=1)
            self.y = self.df['Price']
        else:
            raise ValueError("'Price' column not found in dataset")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        print(f"✓ Training set size: {self.X_train.shape}")
        print(f"✓ Testing set size: {self.X_test.shape}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print(f"✓ Features scaled using StandardScaler")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_linear_regression(self):
        """Train Linear Regression model"""
        print("\n" + "-" * 70)
        print("Training Linear Regression Model...")
        print("-" * 70)
        
        model = LinearRegression()
        model.fit(self.X_train_scaled, self.y_train)
        
        # Predictions
        y_pred_train = model.predict(self.X_train_scaled)
        y_pred_test = model.predict(self.X_test_scaled)
        
        # Evaluation
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        
        print(f"\n✓ Model trained successfully!")
        print(f"\nPerformance Metrics:")
        print(f"  Training R² Score: {train_r2:.4f}")
        print(f"  Testing R² Score: {test_r2:.4f}")
        print(f"  Training RMSE: ${train_rmse:,.2f}")
        print(f"  Testing RMSE: ${test_rmse:,.2f}")
        print(f"  Testing MAE: ${test_mae:,.2f}")
        
        self.models['Linear Regression'] = {
            'model': model,
            'predictions': y_pred_test,
            'r2': test_r2,
            'rmse': test_rmse,
            'mae': test_mae
        }
        
        return model
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "-" * 70)
        print("Training Random Forest Model...")
        print("-" * 70)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(self.X_train_scaled, self.y_train)
        
        # Predictions
        y_pred_train = model.predict(self.X_train_scaled)
        y_pred_test = model.predict(self.X_test_scaled)
        
        # Evaluation
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        
        print(f"\n✓ Model trained successfully!")
        print(f"\nPerformance Metrics:")
        print(f"  Training R² Score: {train_r2:.4f}")
        print(f"  Testing R² Score: {test_r2:.4f}")
        print(f"  Training RMSE: ${train_rmse:,.2f}")
        print(f"  Testing RMSE: ${test_rmse:,.2f}")
        print(f"  Testing MAE: ${test_mae:,.2f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop 5 Most Important Features:")
        print(feature_importance.head())
        
        self.models['Random Forest'] = {
            'model': model,
            'predictions': y_pred_test,
            'r2': test_r2,
            'rmse': test_rmse,
            'mae': test_mae,
            'feature_importance': feature_importance
        }
        
        return model
    
    def train_gradient_boosting(self):
        """Train Gradient Boosting model"""
        print("\n" + "-" * 70)
        print("Training Gradient Boosting Model...")
        print("-" * 70)
        
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        
        # Predictions
        y_pred_train = model.predict(self.X_train_scaled)
        y_pred_test = model.predict(self.X_test_scaled)
        
        # Evaluation
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        
        print(f"\n✓ Model trained successfully!")
        print(f"\nPerformance Metrics:")
        print(f"  Training R² Score: {train_r2:.4f}")
        print(f"  Testing R² Score: {test_r2:.4f}")
        print(f"  Training RMSE: ${train_rmse:,.2f}")
        print(f"  Testing RMSE: ${test_rmse:,.2f}")
        print(f"  Testing MAE: ${test_mae:,.2f}")
        
        self.models['Gradient Boosting'] = {
            'model': model,
            'predictions': y_pred_test,
            'r2': test_r2,
            'rmse': test_rmse,
            'mae': test_mae
        }
        
        return model
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        
        comparison_df = pd.DataFrame({
            'Model': self.models.keys(),
            'R² Score': [m['r2'] for m in self.models.values()],
            'RMSE': [m['rmse'] for m in self.models.values()],
            'MAE': [m['mae'] for m in self.models.values()]
        })
        
        print(f"\n{comparison_df.to_string(index=False)}")
        
        # Find best model
        best_model = comparison_df.loc[comparison_df['R² Score'].idxmax()]
        print(f"\n✓ Best Model: {best_model['Model']} (R² = {best_model['R² Score']:.4f})")
        
        return comparison_df
    
    def plot_actual_vs_predicted(self, save_path='actual_vs_predicted.png'):
        """Plot actual vs predicted prices for all models"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Actual vs Predicted Car Prices', fontsize=16, fontweight='bold')
        
        for idx, (model_name, model_data) in enumerate(self.models.items()):
            ax = axes[idx]
            predictions = model_data['predictions']
            
            ax.scatter(self.y_test, predictions, alpha=0.6, color='#3498DB')
            ax.plot([self.y_test.min(), self.y_test.max()], 
                   [self.y_test.min(), self.y_test.max()], 
                   'r--', lw=2, label='Perfect Prediction')
            
            ax.set_xlabel('Actual Price ($)', fontsize=11)
            ax.set_ylabel('Predicted Price ($)', fontsize=11)
            ax.set_title(f'{model_name}\n(R² = {model_data["r2"]:.4f})', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved as '{save_path}'")
        plt.show()
    
    def plot_residuals(self, save_path='residuals.png'):
        """Plot residuals for all models"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Residual Analysis', fontsize=16, fontweight='bold')
        
        for idx, (model_name, model_data) in enumerate(self.models.items()):
            ax = axes[idx]
            predictions = model_data['predictions']
            residuals = self.y_test - predictions
            
            ax.scatter(predictions, residuals, alpha=0.6, color='#E74C3C')
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel('Predicted Price ($)', fontsize=11)
            ax.set_ylabel('Residuals ($)', fontsize=11)
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved as '{save_path}'")
        plt.show()
    
    def plot_feature_importance(self, save_path='feature_importance.png'):
        """Plot feature importance from Random Forest"""
        if 'Random Forest' in self.models and 'feature_importance' in self.models['Random Forest']:
            feature_imp = self.models['Random Forest']['feature_importance']
            
            plt.figure(figsize=(10, 6))
            plt.barh(feature_imp['Feature'], feature_imp['Importance'], color='#2ECC71')
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.title('Feature Importance in Random Forest Model', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved as '{save_path}'")
            plt.show()
    
    def predict_new_car(self, features_dict):
        """Predict price for a new car"""
        print("\n" + "=" * 70)
        print("PREDICTING NEW CAR PRICE")
        print("=" * 70)
        
        # Create feature array
        features = np.array([features_dict[col] for col in self.X.columns]).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        print(f"\nCar Features: {features_dict}")
        print(f"\nPredictions from different models:")
        
        predictions = {}
        for model_name, model_data in self.models.items():
            model = model_data['model']
            pred_price = model.predict(features_scaled)[0]
            predictions[model_name] = pred_price
            print(f"  {model_name}: ${pred_price:,.2f}")
        
        # Average prediction
        avg_price = np.mean(list(predictions.values()))
        print(f"\n✓ Average Prediction: ${avg_price:,.2f}")
        
        return predictions
    
    def generate_report(self):
        """Generate complete analysis report"""
        print("\n" + "=" * 70)
        print("CAR PRICE PREDICTION - COMPLETE REPORT")
        print("=" * 70)
        
        self.display_dataset_info()
        self.analyze_features()
        self.preprocess_data()
        
        print("\n" + "=" * 70)
        print("MODEL TRAINING")
        print("=" * 70)
        
        self.train_linear_regression()
        self.train_random_forest()
        self.train_gradient_boosting()
        
        self.compare_models()
        
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS...")
        print("=" * 70)
        
        self.plot_actual_vs_predicted()
        self.plot_residuals()
        self.plot_feature_importance()
        
        print("\n" + "=" * 70)
        print("REPORT GENERATION COMPLETE!")
        print("=" * 70)


if __name__ == "__main__":
    # Initialize the model
    model = CarPricePredictionModel('car_price_data.csv')
    
    # Generate complete report
    model.generate_report()
    
    # Example: Predict price for a new car
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    
    # Get feature names from dataset
    feature_names = [col for col in model.df.columns if col != 'Price']
    
    # Create a sample car (you can modify these values)
    sample_car = {feature_names[0]: 100 for feature_names[0:]}  # Placeholder
    
    # Uncomment below to make actual prediction
    # predictions = model.predict_new_car(sample_car)
