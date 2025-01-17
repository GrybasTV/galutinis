
# Euroleague Game Prediction Project

## Tikslas
Naudojant turimus Eurolygos box score duomenis, prognozuoti būsimų rungtynių rezultatus.

## Projekto Apžvalga
Šis projektas skirtas prognozuoti Eurolygos rungtynių rezultatus naudojant LSTM ir GRU modelius, kurie apdoroja laiko sekas. Modeliai treniruojami su surinktais ir apdorotais Eurolygos box score duomenimis, kad būtų galima prognozuoti komandos taškų skaičių ateinančiose rungtynėse. Projekto tikslas yra sukurti patikimą sistemą, kuri gali prognozuoti rungtynių rezultatus remiantis komandos pasirodymu.

## Failų Apžvalga

- `euroleagueapi.py`: Duomenų surinkimas iš Eurolygos API.  
- `preprocess.py`: Duomenų paruošimas ir apdorojimas, įskaitant trūkstamų duomenų užpildymą, agregavimą ir normalizavimą.  
- `train_model.py`: Modelių treniravimas naudojant LSTM ir GRU tinklus.  
- `final.py`: Naudojama Streamlit UI aplikacija, kuri leidžia vartotojui pasirinkti komandas ir gauti prognozes apie rungtynių rezultatus.  
- `README.md`: Projekto aprašymas ir naudojimo instrukcijos.


## Naudos bibliotekos 
tensorflow
numpy
pandas
scikit-learn
matplotlib
streamlit
Euroleague_api

## Duomenų Rinkinys

Projekto metu naudojamas Eurolygos box score duomenų rinkinys, kurio kiekvienas įrašas apima:
- Sezoną, žaidimo kodą, komandą, žaidėjo informaciją, pelnytus taškus, rezultatyvius perdavimus, perimtus kamuolius, klaidas ir kitus statistikos rodiklius.


Duomenys pateikiami CSV formatu ir yra naudojami modelio treniravimui ir vertinimui.

### Pavyzdys:

Duomenis ir jų reikšmės:
Pvz: 
2024,189,1,P007056   ,1.0,1.0,MUN,31,"BOOKER, DEVIN",32:05,17,0,1,4,6,5,6,1,2,3,2,0,1,0,0,2,7,22,0.0,189

2024 – Sezonas. Tai nurodo, kurių metų Eurolygos sezonas.
189 – Žaidimo kodas. Tai unikalus identifikatorius, susijęs su rungtynėmis.
1 – Rungtynių namų komandos indikatorius (Home). Jei reikšmė yra 1, komanda žaidžia namuose; jei 0, tai yra svečių komanda.
P007056 – Žaidėjo ID. Unikalus identifikatorius, kuris nurodo konkretų žaidėją.
1.0 – Ar žaidėjas pradėjo žaidimą kaip starto penketo narys (IsStarter). 1.0 reiškia, kad pradėjo, o 0.0 – kad ne.
1.0 – Ar žaidėjas žaidė rungtynėse (IsPlaying). 1.0 reiškia, kad žaidė, o 0.0 – kad nežaidė.
MUN – Komandos trumpinys. Pvz., „MUN“  Miuncheno „Bayern“ komanda.
31 – Žaidėjo numeris (Dorsal). Marškinėlių numeris.
"BOOKER, DEVIN" – Žaidėjo vardas ir pavardė.
32:05 – Žaidimo minutės (Minutes). Kiek laiko žaidėjas praleido aikštelėje.
17 – Pelnyti taškai (Points). Žaidėjo pelnytų taškų skaičius.
0 – Dviejų taškų metimų pataikymai (FieldGoalsMade2).
1 – Dviejų taškų metimų bandymai (FieldGoalsAttempted2).
4 – Trijų taškų metimų pataikymai (FieldGoalsMade3).
6 – Trijų taškų metimų bandymai (FieldGoalsAttempted3).
5 – Baudų pataikymai (FreeThrowsMade).
6 – Baudų bandymai (FreeThrowsAttempted).
1 – Atkovoti kamuoliai puolime (OffensiveRebounds).
2 – Atkovoti kamuoliai gynyboje (DefensiveRebounds).
3 – Bendras atkovotų kamuolių skaičius (TotalRebounds).
2 – Rezultatyvūs perdavimai (Assistances).
0 – Perimti kamuoliai (Steals).
1 – Kamuolio praradimai (Turnovers).
0 – Blokai už (BlocksFavour). Blokai, kuriuos žaidėjas atliko varžovams.
0 – Blokai prieš (BlocksAgainst). Blokai, kuriuos žaidėjas patyrė nuo varžovų.
2 – Asmeninės pražangos (FoulsCommited).
7 – Priverstos pražangos (FoulsReceived). Kiek kartų varžovai pažeidė taisykles prieš žaidėją.
22 – Efektyvumo koeficientas (Valuation). Bendra žaidėjo žaidimo kokybės vertė pagal Eurolygos formulę.
0.0 – „+/-“ rodiklis (Plusminus). Žaidėjo buvimo aikštelėje įtaka komandos rezultatui (teigiamas arba neigiamas poveikis komandai, kol jis buvo aikštėje).
189 – Pakartotinis žaidimo kodas (Gamecode). Papildomas rungtynių identifikatorius.



## Modelio Architektūra

### LSTM ir GRU
Projekte buvo naudojami LSTM ir GRU modeliai, nes jie puikiai tinka laiko sekų duomenų analizei. Abi architektūros buvo išbandytos su įvairiais hiperparametrais, siekiant nustatyti, kuri iš jų geriausiai prognozuoja Eurolygos rungtynių rezultatus.

### Modelių Treniruotė
Modeliai buvo treniruojami naudojant 80% duomenų rinkinio, o likusi dalis buvo skirta validacijai. Kiekvienam lango dydžiui (1-10 žaidimų) buvo sukurti skirtingi modeliai.

### Įvertinimas
Modelių veikimas buvo vertinamas naudojant Mean Absolute Error (MAE), F1 balą, ir ROC AUC metrikas. Geriausi modeliai buvo išsaugoti ir gali būti naudojami prognozėms.

## Naudojimas

1. **Duomenų Surinkimas**  
   Norėdami pradėti, sukurkite `data/euroleague_boxscores_all.csv` failą, kuriame bus saugomi visi surinkti duomenys. Naudokite `euroleagueapi.py` failą duomenų gavimui.

2. **Duomenų Apdorojimas**  
   Paleiskite `preprocess.py` skriptą, kad apdorotumėte ir paruoštumėte duomenis modelio treniravimui. Skriptas sukurs laiko sekas, normalizuos ir suskirstys duomenis į treniravimo ir validavimo rinkinius.

3. **Modelio Treniruotė**  
   Treniruokite modelį naudodami `train_model.py` failą. Šiame faile aprašyta LSTM ir GRU modelių treniravimo logika, taip pat įvertinimas pagal įvairius langų dydžius.

4. **Prognozės su Streamlit UI**  
   Naudokite `final.py` failą, kad paleistumėte Streamlit aplikaciją. Tai leis jums pasirinkti dvi komandas ir pamatyti prognozuojamą rezultatą, taip pat kitus statistinius duomenis apie komandas.

## Pavyzdys, kaip naudoti:

1. Paleiskite `train_model.py` ir palaukite, kol modeliai bus treniruojami.
2. Paleiskite `final.py` ir pasirinkite dvi komandas iš sąrašo.
3. Aplikacija parodys prognozuojamus taškus ir tikimybę, kad kiekviena komanda laimės.

## Ateities Patobulinimai

- Išbandyti sudėtingesnes modelių architektūras, pavyzdžiui, dėmesio mechanizmus (Attention Mechanisms).
- Integruoti papildomus duomenis, tokius kaip žaidėjų individualios statistikos duomenys.
- Atlikti hiperparametrų optimizavimą naudojant Grid Search arba Bayesian Optimization metodus.

## Licencija

Šis projektas yra atviras šaltinis, ir jūs galite naudoti bei modifikuoti kodą pagal savo poreikius. Jei naudojate šį projektą komerciniams tikslams, nurodykite šaltinį.

---

This README provides an overview of your project, the file structure, and how to run the models to get predictions. Feel free to adjust the details if needed!





