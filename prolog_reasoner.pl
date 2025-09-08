% Predicati per le feature normalizzate e gli indici di colore
u(Stella, Val) :- magnitude(Stella, u, Val).
g(Stella, Val) :- magnitude(Stella, g, Val).
r(Stella, Val) :- magnitude(Stella, r, Val).
i(Stella, I) :- magnitude(Stella, i, I).
z(Stella, Z) :- magnitude(Stella, z, Z).

% Regole per gli indici di colore (corrette)
ug(Stella, Indice) :- u(Stella, U), g(Stella, G), Indice is U - G.
gr(Stella, Indice) :- g(Stella, G), r(Stella, R), Indice is G - R.
ri(Stella, Indice) :- r(Stella, R), i(Stella, I), Indice is R - I.
iz(Stella, Indice) :- i(Stella, I), z(Stella, Z), Indice is I - Z.

% Predicato di ragionamento principale con spiegazione (versioni uniche e corrette)
% NOTA: L'ordine delle regole è importante.

risolvi_ambiguita(Stella, 'QSO', 'Il suo indice u-g e'' basso, tipico di un Quasar.') :-
    ug(Stella, IndiceUG), gr(Stella, IndiceGR),
    IndiceUG =< 0.5,
    IndiceGR < 0.7, !.

risolvi_ambiguita(Stella, 'STAR', 'Il suo indice u-g e'' alto e r-i e'' basso, tipico di una Stella.') :-
    ug(Stella, IndiceUG), gr(Stella, IndiceGR), ri(Stella, IndiceRI),
    IndiceUG > 0.5,
    IndiceGR > -0.2,
    IndiceRI =< 0.6, !.

risolvi_ambiguita(Stella, 'GALAXY', 'I suoi indici g-r e r-i sono alti, tipico di una Galassia.') :-
    gr(Stella, IndiceGR), ri(Stella, IndiceRI),
    IndiceGR > 0.7,
    IndiceRI > 0.3, !.


risolvi_ambiguita(Stella, 'RED_DWARF', 'Il suo indice r-i e'' molto alto, tipico di una Nana Rossa.') :-
    gr(Stella, IndiceGR),
    ri(Stella, IndiceRI),
    IndiceGR < 0.2,
    IndiceRI > 0.8, !.

risolvi_ambiguita(Stella, 'WHITE_DWARF', 'Il suo indice g-r e'' negativo, tipico di una Nana Bianca.') :-
    gr(Stella, IndiceGR),
    IndiceGR < 0, !.
% Predicato di fallback se nessuna regola si applica
risolvi_ambiguita(_, 'Indefinito', 'Nessuna regola si applica per questo oggetto.').