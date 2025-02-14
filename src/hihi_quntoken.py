import quntoken


test_text = """
Élt egyszer messze keleten egy nagy királyságban egy bátor és jószívű király. A királyságban mindenki szerette a királyt, mert igazságosan uralkodott, és mindig megvédte alattvalóit. A király azonban egy nagy titkot őrzött: titokban krampampuli király is volt, egy hatalmas erdő mélyén, ahol a legfélelmetesebb szörnyek laktak. Ebben az erdőben élt a krampampuli király, aki igazságtalanul és kegyetlenül uralkodott alattvalói felett.

Egy napon a messzi királyságban a király egy furcsa levelet kapott, amelyben azt írták, hogy a krampampuli király megtámadta az erdei állatokat, és kincseket rabolt el tőlük. A király felháborodott, és elhatározta, hogy meglátogatja a krampampuli király birodalmát, és megállítja őt gonosz tetteiben.

Amikor a messzi királyság királya megérkezett a krampampuli király birodalmába, megdöbbent, milyen pusztítást végeztek a szörnyek. Az erdőt kifosztották, az állatokat pedig elhurcolták. A király azonban nem adta fel, és elindult, hogy megkeresse a szörnyek királyát.

Végül egy barlang mélyére ért, ahol megtalálta a krampampuli királyt. A barlangban volt egy hatalmas tó, amelyben egy furcsa lény, egy krampampuli élt. A krampampuli király kegyetlenül bánt alattvalóival, és mindenkit elrabolt, aki az erdőben élt.

Amikor a messzi királyság királya meglátta a krampampuli királyt, nagyon megdöbbent. De bátor volt, és nem hátrált meg. Megkérdezte a krampampuli királyt, hogy miért viselkedik ilyen kegyetlenül alattvalóival.

A krampampuli király azonban csak nevetett, és azt mondta, hogy ő a leghatalmasabb lény a földön. A messzi királyság királya azonban nem adta fel, és egy csodálatos varázslatot mondott a krampampuli királyra. A varázslat hatására a krampampuli király elvesztette hatalmát, és bűnbánóan megtért.

A messzi királyság királya visszatért birodalmába, és elmondta alattvalóinak, hogyan győzte le a gonosz krampampuli királyt. A királyságban mindenki nagyon hálás volt a messzi királynak, aki megmentette őket a gonosztól. A krampampuli király pedig megfogadta, hogy soha többé nem tér vissza gonosz tetteihez, és hálásan élte le hátralévő napjait.
"""

import io
import sys
try:
    import quntoken
except ImportError:
    print("Error: quntoken package is not installed. Please install it using:")
    print("pip install quntoken")
    sys.exit(1)

try:
    res = quntoken.tokenize(io.StringIO(test_text), form='raw', mode='sentence', word_break=False)
    for line in res:
        print(line)
except Exception as e:
    print(f"Error occurred while tokenizing: {str(e)}")
    sys.exit(1)
