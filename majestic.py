import telebot
import numpy as np
from collections import Counter
import pandas as pd
import os
import time
from telebot import types
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Список разрешенных пользователей
ALLOWED_USERS = []

class RoulettePredictor:
    def __init__(self, data):
        # Определение цветов чисел
        self.black_numbers = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}
        self.red_numbers = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
        self.green_numbers = {0, '00'}
        
        # Преобразование данных, сохраняя различие между 0 и 00
        self.data = []
        for x in data.replace(";", ",").split(","):
            if x:
                if x == '00':
                    self.data.append('00')
                else:
                    self.data.append(int(x))
                    
        self.last_numbers = {}
        self.sequence_stats = {}
        self.total_numbers = len(self.data)
        self._build_sequences()
        self._calculate_global_stats()

        self.model = None
        self.scaler = StandardScaler()
        self.ml_features = None
        self.ml_targets = None
        self.window_size = 5
        self._prepare_data_for_ml()
        self._train_ml_model()
    

    def _get_color(self, number):
        if number in self.black_numbers:
            return 'Черное'
        elif number in self.red_numbers:
            return 'Красное'
        else:
            return 'Зеленое'

    def _build_sequences(self):
        self.last_numbers = {}
        self.sequence_stats = {}
        
        for i in range(len(self.data) - 1):
            current_num = self.data[i]
            next_num = self.data[i + 1]
            
            if current_num not in self.last_numbers:
                self.last_numbers[current_num] = []
                self.sequence_stats[current_num] = {'total': 0, 'next_numbers': {}}
            
            self.last_numbers[current_num].append(next_num)
            self.sequence_stats[current_num]['total'] += 1
            
            if next_num not in self.sequence_stats[current_num]['next_numbers']:
                self.sequence_stats[current_num]['next_numbers'][next_num] = 0
            self.sequence_stats[current_num]['next_numbers'][next_num] += 1

    def _calculate_global_stats(self):
        self.global_stats = Counter(self.data)
        self.number_probabilities = {
            num: count / self.total_numbers * 100 
            for num, count in self.global_stats.items()
        }

    def get_number_probability(self, number):
        return self.number_probabilities.get(number, 0)

    def _create_features(self, last_numbers):
      features = []
      
      last_numbers = [int(x) if isinstance(x, str) and x != '00' else 37 if x=='00' else x for x in last_numbers]

      last_numbers = last_numbers[-self.window_size:]
      
      padding_length = self.window_size - len(last_numbers)
      features.extend([37] * padding_length)
      features.extend(last_numbers)
      
      return np.array(features).reshape(1, -1)

    def _prepare_data_for_ml(self):
        if len(self.data) < self.window_size + 1:
            self.ml_features = None
            self.ml_targets = None
            return

        features = []
        targets = []
        ml_data = [37 if x == '00' else x for x in self.data if x != '']
        for i in range(len(ml_data) - self.window_size):
            features.append(ml_data[i:i+self.window_size])
            targets.append(ml_data[i+self.window_size])
        
        self.ml_features = np.array(features)
        self.ml_targets = np.array(targets)

    def _train_ml_model(self):
            if self.ml_features is None or self.ml_targets is None or self.ml_features.size == 0 or self.ml_targets.size == 0:
                return

            X_train, X_test, y_train, y_test = train_test_split(
                self.ml_features, self.ml_targets, test_size=0.2, random_state=42, stratify=self.ml_targets
                )

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)


            best_accuracy = 0
            best_model = None
        
            models = {
                'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42, learning_rate=0.1, n_estimators=100, max_depth=3),
                'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', learning_rate=0.1, n_estimators=200, max_depth=3)
            }

            for name, model in models.items():
                try:
                     model.fit(X_train_scaled, y_train)
                     y_pred = model.predict(X_test_scaled)
                     accuracy = accuracy_score(y_test, y_pred)

                     if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model
                except Exception as e:
                     print(f"Error with {name}: {e}")

            if best_model is not None:
                 self.model = best_model
                 print(f"Best ML model selected with accuracy: {best_accuracy*100:.2f}%")
            else:
                  print("No suitable ML model could be trained.")
    
    def predict_with_ml(self, last_numbers, top_n=3):
        if self.model is None or not last_numbers:
            return self.predict_frequent_numbers(top_n)
        
        features = self._create_features(last_numbers)
        features_scaled = self.scaler.transform(features)
        
        probas = self.model.predict_proba(features_scaled)[0]
        predictions_with_proba = list(zip(self.model.classes_, probas))
        sorted_predictions = sorted(predictions_with_proba, key=lambda x: x[1], reverse=True)[:top_n]
        
        return sorted_predictions


    def predict_next_numbers(self, last_number, top_n=3):
        if last_number not in self.sequence_stats:
            return self.predict_frequent_numbers(top_n)

        stats = self.sequence_stats[last_number]
        total_sequences = stats['total']
        next_numbers_stats = stats['next_numbers']

        probabilities = {
            num: (count / total_sequences * 100)
            for num, count in next_numbers_stats.items()
        }

        sorted_predictions = sorted(
            probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        return sorted_predictions

    def predict_frequent_numbers(self, top_n=3):
        return [(num, self.get_number_probability(num)) 
                for num, _ in self.global_stats.most_common(top_n)]

    def analyze_patterns(self):
        color_counts = {'Черное': 0, 'Красное': 0, 'Зеленое': 0}
        for num in self.data:
            if num == '00' or num == 0:
                color_counts['Зеленое'] += 1
            elif num in self.black_numbers:
                color_counts['Черное'] += 1
            elif num in self.red_numbers:
                color_counts['Красное'] += 1

        color_stats = {
            color: (count / self.total_numbers * 100)
            for color, count in color_counts.items()
        }

        numeric_data = [x for x in self.data if x != '00']
        df = pd.DataFrame(numeric_data, columns=['number'])
        
        even_odd = df['number'].apply(lambda x: 'Чётное' if x % 2 == 0 else 'Нечётное')
        even_odd_stats = even_odd.value_counts(normalize=True) * 100
        
        def get_range(n):
            if n == 0: return '0'
            if 1 <= n <= 12: return '1-12'
            if 13 <= n <= 24: return '13-24'
            return '25-36'
        
        range_stats = df['number'].apply(get_range).value_counts(normalize=True) * 100
        
        zero_double_count = self.data.count('00')
        zero_double_percent = (zero_double_count / self.total_numbers) * 100
        
        return {
            'colors': color_stats,
            'even_odd': even_odd_stats.to_dict(),
            'ranges': range_stats.to_dict(),
            'zero_double': zero_double_percent
        }

    def add_new_data(self, new_number):
        self.data.append(new_number)
        self.total_numbers += 1
        self._build_sequences()
        self._calculate_global_stats()
        self._prepare_data_for_ml()
        self._train_ml_model()

    def save_predictions_to_file(self, predictions, ml_predictions, patterns, number):
        with open("MajesticPredictions.txt", "a") as f:
            f.write(f"\nПрогноз после числа {number}:\n")
            f.write("-" * 40 + "\n")
            
            f.write("На основе последовательности:\n")
            for predicted_number, probability in predictions:
                f.write(f"Число {predicted_number}: {probability:.2f}% вероятность\n")
            
            f.write("\nНа основе ML модели:\n")
            for predicted_number, probability in ml_predictions:
                f.write(f"Число {predicted_number}: {probability*100:.2f}% вероятность\n")
            
            f.write("\nТекущая статистика:\n")
            f.write("-" * 40 + "\n")
            
            for color, percent in patterns['colors'].items():
                f.write(f"{color}: {percent:.2f}%\n")
            
            f.write("\nЧётные/Нечётные:\n")
            for category, percent in patterns['even_odd'].items():
                f.write(f"{category}: {percent:.2f}%\n")
            
            f.write("\nДиапазоны:\n")
            for range_name, percent in sorted(patterns['ranges'].items()):
                if range_name == '0':
                    f.write(f"0: {percent:.2f}%\n")
                else:
                    f.write(f"{range_name}: {percent:.2f}%\n")
            
            f.write(f"\nВероятность '00': {patterns['zero_double']:.2f}%\n")
            f.write("-" * 40 + "\n")

bot = telebot.TeleBot('your_token')

initial_data = "8;24;22;32;2;23;11;29;9;8;16;1;18;34;1;14;5;34;25;7;8;26;2;10;5;4;32;29;5;10;20;8;35;13;15;29;11;4;29;1;14;33;31;13;11;3;8;23;4;11;26;29;18;00;31;7;31;1;20;22;8;34;10;6;4;15;4;24;4;6;34;11;24;9;22;27;34;00;20;9;25;20;11;27;20;6;24;9;5;35;35;14;35;5;3;0;13;30;24;14;25;17;25;26;16;2;26;34;11;17;11;11;4;22;15;28;33;17;22;20;10;16;0;3;14;27;4;12;24;10;19;2;6;11;4;29;30;10;35;12;18;34;32;6;30;12;24;12;1;14;12;2;5;28;24;19;31;36;19;32;14;21;22;2;14;29;8;6;31;29;6;15;11;8;2;21;14;30;6;11;31;31;16;12;25;00;18;34;16;6;32;34;1;0;17;25;23;33;35;6;36;4;27;23;30;34;15;14;13;30;10;10;14;9;5;31;36;27;18;7;12;3;17;26;11;4;18;32;13;35;19;15;13;11;24;16;18;11;31;3;28;20;5;22;32;32;0;32;30;27;30;29;35;24;23;30;26;15;32;32;29;11;3;18;18;23;9;17;36;12;9;31;15;31;28;36;7;13;16;19;6;20;21;25;21;25;33;3;12;33;0;27;7;10;18;12;17;28;23;2;1;17;36;33;33;26;32;30;5;24;33;12;22;19;"

predictors = {}


authorized_users = set()

@bot.message_handler(commands=['start'])
def start(message):
    user_id = message.from_user.id
    
    if user_id in ALLOWED_USERS:
        authorized_users.add(user_id)
        
        markup = types.ReplyKeyboardMarkup(row_width=2)
        item1 = types.KeyboardButton("Majestic RP")
        item2 = types.KeyboardButton("GTA5RP")
        item3 = types.KeyboardButton("Radmir CRMP")
        item4 = types.KeyboardButton("Radmir GTA v")
        markup.add(item1, item2, item3, item4)
        
        bot.reply_to(message, 
                     "Здравствуйте! Spin Predictor - единственный многопроектный чит для предсказания следующего числа в рулетке. Выберите ваш проект:", 
                     reply_markup=markup)
    else:
        bot.reply_to(message, "Доступ запрещен")

@bot.message_handler(func=lambda message: message.text in ["Majestic RP", "GTA5RP", "Radmir CRMP", "Radmir GTA v"])
def handle_project_choice(message):
    if message.text == "Majestic RP":
        chat_id = message.chat.id
        predictors[chat_id] = RoulettePredictor(initial_data)
        bot.reply_to(message, 
                     "Проект Majestic RP выбран. Отправьте число от 0 до 36 или '00', и я предоставлю статистику и прогноз.")
    else:
        bot.reply_to(message, "Извините, эта функция находится в разработке")

@bot.message_handler(func=lambda message: True)
def handle_number(message):
    user_id = message.from_user.id
    
    if user_id not in ALLOWED_USERS:
        bot.reply_to(message, "Доступ запрещен")
        return
        
    chat_id = message.chat.id
    
    if chat_id not in predictors:
        bot.reply_to(message, "Пожалуйста, выберите проект с помощью кнопок.")
        return
    
    try:
        input_number = message.text.strip()
        
        if input_number == '00':
            number = '00'
        else:
            number = int(input_number)
            if not (0 <= number <= 36):
                raise ValueError()

        predictor = predictors[chat_id]
        
        predictor.add_new_data(number)
        
        sequence_predictions = predictor.predict_next_numbers(number)
        ml_predictions = predictor.predict_with_ml(predictor.data[-3:])
        patterns = predictor.analyze_patterns()
        
        predictor.save_predictions_to_file(sequence_predictions, ml_predictions, patterns, number)

        response = f"📊 Прогноз после числа {number}:\n\n"

        response += "На основе последовательности:\n"
        for predicted_number, probability in sequence_predictions:
            response += f"Число {predicted_number}: {probability:.2f}% вероятность\n"
        
        response += "\nНа основе ML модели:\n"
        for predicted_number, probability in ml_predictions:
            response += f"Число {predicted_number}: {probability*100:.2f}% вероятность\n"
        
        response += "\n📈 Текущая статистика:\n"
        
        response += "\nЦвета:\n"
        for color, percent in patterns['colors'].items():
            response += f"{color}: {percent:.2f}%\n"
        
        response += "\nЧётные/Нечётные:\n"
        for category, percent in patterns['even_odd'].items():
            response += f"{category}: {percent:.2f}%\n"
        
        response += "\nДиапазоны:\n"
        for range_name, percent in sorted(patterns['ranges'].items()):
             if range_name == '0':
                response += f"0: {percent:.2f}%\n"
             else:
                response += f"{range_name}: {percent:.2f}%\n"
        
        response += f"\nВероятность '00': {patterns['zero_double']:.2f}%"

        bot.reply_to(message, response)

    except ValueError:
        bot.reply_to(message, "Пожалуйста, введите число от 0 до 36 или '00'")
    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    print("Бот запущен...")
    bot.infinity_polling()
