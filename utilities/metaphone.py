from itertools import chain, product

'''
1: а, о, у, ы
3: и, е, ё, э, ю
4(1,3): я
5: б, п
6: в, ф
7: д, т
8: г, к, х
9: л
10: м
11: н
12: р
13: з, с
14: й
15: щ, ч
16: ш, ж
17: ц
18: ь,ъ
'''

SIBILANTS = ['з', 'с', 'щ', 'ч', 'ш', 'ц', 'ж']
VOWELS = {'а': 1, 'я': 4, 'о': 1, 'ё': 3, 'э': 3, 'е': 3, 'у': 1, 'ю': 3, 'ы': 1, 'и': 3}
SIMPLE_CONSONANTS = {'б': 5, 'п': 5, 'в': 6, 'ф': 6, 'г': 8, 'к': 8,
                     'х': 8, 'р': 12, 'л': 9, 'м': 10, 'н': 11}
SIGNS = {'ь', 'ъ'}


def transform(word):
    word = word.lower()
    state = 0
    answer = []
    for letter in word:
        if isinstance(state, tuple):
            # переводим пару состояний, возникшую в случае мягкого знака, в одно состояние
            if letter in VOWELS:
                # для последующего гласного важна только мягкость
                state = state[1]
            else:
                # для согласного --- только тип предыдущего согласного
                state = state[0]
        # печатаем буквы, сохранённые в состояниях и не выданные в ответ
        if state == 7:  # предыдущая буква 'т', 'д'
            if letter not in ['з', 'с', 'ц', 'ч', 'щ'] and letter not in SIGNS:
                answer.append(7)
        elif state in [13, 16] and letter not in SIGNS:
            if letter not in SIBILANTS:
                answer.append(state)
        elif state == 14:
            if letter not in VOWELS:
                answer.append(14)
        elif state == 17  and letter not in SIGNS:
            if letter not in ['щ', 'ч']:
                answer.append(17)
        # обрабатываем следующую букву
        if letter in VOWELS:
            if state in [1, 3, 4]:
                continue
            if state not in [1, 3, 4, 14, 15, 16, 17, 18]:
                new_state = VOWELS[letter]
            elif state in [14, 15, 18]:
                new_state = 3
            elif state in [16, 17]:
                new_state = 1
            answer.append(new_state)
            state = new_state
        elif letter in SIMPLE_CONSONANTS:
            new_state = SIMPLE_CONSONANTS[letter]
            if state != new_state:
                answer.append(new_state)
                state = new_state
        elif letter in ['д', 'т']:
            state = 7
        elif letter in ['з', 'с']:
            if state == 7:
                state = 17
            else:
                state = 13
        elif letter == 'й':
            state = 14
        elif letter in ['щ', 'ч']:
            if state != 15:
                answer.append(15)
                state = 15
        elif letter in ['ш', 'ж']:
            state = 16
        elif letter in ['ц']:
            state = 17
        elif letter in ['ь', 'ъ']:
            # сохраняем предыдущее состояние и добавляем индикатор мягкости
            state = (state, 18)
        elif letter == ['-']:
            state = 0
    # печатаем буквы, сохранённые в состояниях и не выданные в ответ
    if state in [7, 13, 14, 16, 17]:
        answer.append(state)
    variants = [[i] if i != 4 else [1, 3] for i in answer]
    answer = [list(elem) for elem in product(*variants)]
    return answer

if __name__ == "__main__":
    words = ['хотеться', 'хочется', 'хотеца', 'хочеца', 'миллион',
             'мильон', 'вариант', 'варьянт', 'иерей',  'район']
    for word in words:
        print(word, transform(word))

