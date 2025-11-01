# 🏀 Predykcja składów All-NBA i All-Rookie (Python / Machine Learning)

Projekt służy do automatycznej predykcji składów **All-NBA Teams** oraz **All-Rookie Teams** w oparciu o statystyki zawodników NBA z lat 2000–2023.  
Celem było stworzenie modelu, który na podstawie wybranych cech graczy przewidzi ich przynależność do jednej z drużyn wyróżnionych w danym sezonie.

## 📊 Opis danych

Dane źródłowe zostały pobrane z serwisu [basketball-reference.com](https://www.basketball-reference.com/).  
Zawierały 28 statystyk dla każdego zawodnika w jednym sezonie.  
W procesie przygotowania danych wybrano **10 kluczowych cech** (m.in. GS, MIN, FGM, FGA, FTM, FTA, AST, STL, PTS), które posłużyły do uczenia modelu.

Zbiór danych obejmujący sezon **2023/2024** został odłączony od danych treningowych, aby uniknąć sytuacji, w której model przewiduje wyniki na podstawie informacji, które sam widział podczas uczenia.

## 🧠 Trenowanie modelu

Skrypt wczytuje dane z pliku `all_stats.csv`, wykonuje preprocessing oraz trenuje model klasyfikacji.  
Etapy:

1. **Wczytanie danych i czyszczenie**  
   - usunięto kolumny: `Player`, `Year`, `TOV`  
   - zmapowano pozycje zawodników (`F`, `G`, `C`) oraz klasy (`First`, `Second`, `Third`, `not`)  
   - usunięto brakujące wartości  

2. **Skalowanie cech**  
   - zastosowano `StandardScaler` w celu normalizacji wartości wejściowych.  

3. **Trenowanie modelu**  
   - użyto **Logistic Regression** (`max_iter=1000`, `class_weight='balanced'`)  
   - model wytrenowano osobno dla:
     - `All-NBA` (`model_all_nba.joblib`)
     - `All-Rookie` (`model_rookies_nba.joblib`)

4. **Zapis modelu**  
   Modele zapisane przy pomocy biblioteki `joblib` umożliwiają szybkie ponowne wczytanie bez potrzeby ponownego uczenia.

## 🧩 Wybór modelu

Podczas testów porównano różne algorytmy (m.in. Random Forest, SVM).  
Najlepsze wyniki uzyskano dla **regresji logistycznej**, która oferowała stabilną konwergencję i wysoką skuteczność przy niewielkiej liczbie cech.

Model skonfigurowano z:
- `max_iter=1000` – zwiększona liczba iteracji dla pewności konwergencji,  
- `class_weight='balanced'` – automatyczne wyrównanie liczby przykładów z mniejszościowej klasy (zawodnicy wybrani do All-NBA) względem klasy większościowej (pozostali zawodnicy).

## 🔮 Predykcja i generowanie wyników

Etap predykcji wykorzystuje najnowszy sezon (`2023/24`) i generuje końcowe drużyny All-NBA oraz Rookie All-NBA w formacie **JSON**.

### Etapy:
1. Wczytanie danych testowych (`2023_season.csv`, `2023_rookies.csv`).  
2. Wczytanie zapisanych modeli `.joblib`.  
3. Przeprowadzenie predykcji dla obu kategorii.  
4. Wyłonienie składów:
   - **First / Second / Third All-NBA Team**
   - **First / Second Rookie All-NBA Team**
5. Posortowanie zawodników wg punktów i pozycji (2xF, 2xG, 1xC).  
6. Zapis wyników do pliku `.json`.


## ⚙️ Trening modeli

Aby wytrenować modele dla All-NBA oraz All-Rookie, należy uruchomić skrypt Pythona w katalogu projektu.  
Wymagane są pliki danych: `all_stats.csv` oraz `all_rookies.csv`.

W terminalu (np. VS Code) wykonaj:

```bash
python main.py
```

## 🔮 Generowanie predykcji

Po wytrenowaniu modeli możesz wygenerować przewidywane drużyny All-NBA i All-Rookie dla sezonu 2023/2024.
```bash
python main.py wyniki_all_nba.json
```
