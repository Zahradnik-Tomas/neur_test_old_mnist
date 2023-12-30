# neur_test_old_mnist

dal jsem svou dosavadni praci cislo 1, naposledy zmeneno nekdy na zacatku listopadu 2023

dodane vahy jsou jenom "train(5000, a)", takze nikdy nevidely vsechny obrazky (i kdyby to bylo 60 000 tak vsechny vahy neuvidi diky hloupemu zpusobu treninku)
a tim padem je presnost obcas chaoticka (casto by to v softmaxu hazelo aspon 80% na spravnou odpoved, ale taky se to i plete nekdy ... voda je mokra Tome)(taky architektura je docela k nicemu na tento ukol)

soubor "mnist.npz" a zpusob nacitani souboru jsem cmajznul ze sc kerasu

NasobeniMatic, konstruktor a Forward je inspirovano od pana učitele Haberzettela
	-> specificky to ze neuronka je pole matic

Backprop je napsan podle 3 rovnic nachazejicich se na strane 197 v knize "Mobilní Roboty" od Petra Nováka z roku 2005
	-> specificky rovnice (113), (114) a (115)
 
