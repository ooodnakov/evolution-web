# Полная RL-среда для Evolution

> Историческая запись утверждённого плана (2026-10-22). Команды и раскладка файлов ниже
> отражают состояние на момент планирования. Актуальный runtime: Bun 1.4.4 + собранный
> артефакт `dist/rl` (`npm run rl:build`); см. `docs/rl/implementation-notes.md` и
> `docs/commands.md`.

## Context
Построить на существующем JavaScript-движке среду для обучения модели игре в «Эволюцию». Пользователь выбрал конечную поддержку всей существующей игры и 2–8 игроков; базовая игра на двоих — первый проверяемый этап, не сокращение итогового объёма. Результат включает Python multi-agent API, self-play rollout, baseline-соперников и короткую проверку обучения, но не обещание силы модели или полноценную checkpoint-лигу.

## Подтверждённые ограничения
- `shared/actions/ai.js:135-261`: `getOptions` собирает только представителей ходов FEEDING; использует `.find` для целей и не перечисляет DEPLOY/ответы/засады. Нельзя выдавать его результат за полный action mask.
- `shared/utils/reduxTimeout.js:71-103`: таймеры — middleware-контрольные действия с callback-функциями; нулевая duration отменяет постановку, а не выполняет callback. Простое обнуление всех таймеров не заменяет игровой scheduler.
- `shared/models/game/GameModel.js:279-330`: создание игры выбирает колоды дополнений, может перемешивать игроков; `toOthers` маскирует колоду и чужие руки, `toClient` без предварительного `toOthers` не выполняет эту защиту (`:353-369`).
- `GameModel.fromServer` ожидает wire-коллекции и восстанавливает Record-типы (`:332-350`); его нельзя объявлять готовым checkpoint API для произвольного `game.toJS()` без отдельной проверки.

## Зафиксированная стратегия
Переиспользовать правила и authoritative handlers, не переписывать правила на Python. Headless Node-процесс общается с Python через JSON Lines по stdin/stdout; ZeroMQ/HTTP не нужны для локального subprocess. Python предоставляет PettingZoo AEC, поскольку защита/интеллект/засады передают управление не по простому кругу.

Переменные популяции не обрезать: observation использует Gymnasium `Sequence`, действия — авторегрессионный выбор полей с динамическим `Discrete(len(choices))` на каждом префиксе. AEC документирует изменяемые action spaces; объект пространства кешировать в пределах неизменного decision, не пересоздавать на каждом вызове. В training smoke применить candidate-scoring policy, а не фиксированный DQN/PPO head.

Python-зависимости управляются только `uv`: корневые `pyproject.toml`, `uv.lock`, `.python-version`; установка/запуск через `uv sync --locked` / `uv run --locked`. Существующих Python manifest и каталога `rl/` не обнаружено.

## Дополнительные проверенные факты
- `globals.js:5-7` обрезает UUID до 16 бит вне production; коллизии вероятны, но не «гарантированы в любой длинной партии». В env не импортировать `globals.js`.
- `server/database.js` поглощает ошибку соединения через `.catch(console.error)`; прежняя формулировка про неизбежный unhandled rejection была неточна. Импорт Mongo всё равно недопустим в среде.
- `shared/actions/game.js:439-445`: отсутствие turn-timeout автоматически завершает FEEDING после действия. Scheduler должен сохранять логическое присутствие таймера, не просто отключать его.
- `shared/utils/reduxQuestion.js` — отдельная Promise-механика, не используемая игровыми вопросами: поиск `rqActionAsk` нашёл объявления/тесты, не игровой вызов. Не включать её в headless store.
- `server$questionPlantCarnivorousCounterattack`/`QuestionRecord.PLANT_COUNTERATTACK` объявлены, но вызовов фабрики не найдено. Реальная контратака растения идёт через `HUNT_FLAG.PLANT_COUNTERATTACK` и обычную защиту (`traitsData/hunt.js:189-281`); новый prompt для мёртвого объявления не придумывать.
- Hidden PlantCarnivorous/Intellect — встроенные свойства публичного типа `PlantCarnivorous` (`plantTypeData.js:108-117`), а не автоматически секретные возможности. Для политики выводить их из публичного типа; скрытые динамические значения не раскрывать.
- `TraitAnglerfish` имеет `hidden`, `transient`, `score:0`, но без собственного `food` (`traitsData/ttf.js:243-259`). Нельзя утверждать утечку foodSize именно от него без контрпримера; защита наблюдений нужна из-за широкого wire-state, settings.seed, UUID и приватных значений.

## Approach

### 1. Отделить игровые переходы от web-побочных эффектов
Изменения сначала сохраняют web-правила, затем открывают чистый импорт движка. Новые пути этого плана отсутствуют; эквивалента headless API и Python-пакета в репозитории не обнаружено.

1. В `shared/actions/game.js`, `trait.js`, `ai.js`, `shared/models/game/evolution/traitsData/{index,ttf,cons,bonus}.js` заменить игровые импорты barrel `actions/actions` на прямые импорты владельцев: `game`, `trait`, `game.plantarium`, `generic`. Точный перечень обнаруживается поиском `from ['"][^'"]*actions(?:/actions)?['"]` по этим путям, исключая specs. Не переносить правила из traitsData и не создавать вторую реализацию черт.
2. Убрать из `shared/actions/game.js` импорт `db$gameEnd` и зависимость `server$roomSelfDestructStart`. После существующих authoritative `gameEnd` и client-only публикации dispatch нового события `{type:'gameFinished', data:{gameId, finished}}`. В новом `server/middleware/gameLifecycle.js` обработать это событие после `next`: выполнить прежние room-self-destruct, `toDatabase`, `db$gameEnd` и `loggerOnline.info` в прежнем порядке. Подключить middleware в `server/configureStore.js` и web-test store `shared/test-helper.js`. В RL-store подписчика persistence нет по архитектуре; это не мок Mongo и не перехват ошибки соединения.
3. Оставить десериализацию `gameServerToClient` в shared, но убрать оттуда redirect и audio. Добавить `client/configuration/game-handlers.js`: обернуть только `gameInit`, `gameCreateSuccess`, `gameNextPlayer` прежними browser-эффектами; `client/configuration/socket.js` применяет эти overrides поверх registry. В shared больше нет импорта `client/actions/app`. Единственный direct-import consumer `gameServerToClient` — `shared/actions/actions.js`; его pure-map контракт сохраняется.
4. У `shared/utils/logger/transports.js` добавить `LOG_STREAM=stderr`: все уровни Console отправляются в stderr только при этом значении. В `logger/index.js` загружать Spreadsheet transport только при заданном `GOOGLE_LOGS_FILE`; при отсутствии файла transport не нужен. Python worker получает whitelist environment и не наследует Google/Mongo/JWT credentials. Логи не смешивать с JSON-протоколом, но ошибки не подавлять.
5. `globals.js` и `shared/test-helper.js` не входят в production RL import graph. В `shared/test-helper.js` установить тестовый `navigator` через writable/configurable `Object.defineProperty`, поскольку присваивание падает на Node 24; это исправление только совместимости существующего harness, не часть headless bootstrap.

Не менять общий barrel для всех UI-вызовов, auth, комнат и чата: достаточно разорвать конкретные игровые зависимости выше. Удаляемые импорты/эффекты не оставлять параллельными путями.

### 2. Сделать randomness, ID и время контекстом эпизода
Добавить `shared/utils/gameRuntime.js`:
```js
withGameRuntime(runtime, callback)       // synchronous, restores prior context in finally
getGameRuntime()                       // current context or null
gameNow()                              // runtime.now() or Date.now()
gameId(kind)                           // runtime.id(kind) or uuid.v4()
```
Это scoped synchronous service, не monkeypatch `Math.random`, Date или UUID. `withGameRuntime` запрещает Promise-return (`ASYNC_GAME_RUNTIME`); все игровые handlers синхронны, Promise-based `reduxQuestion` в RL не устанавливается. Каждый внешний dispatch, model factory и scheduler callback вызывается внутри originating runtime; вложенные dispatch сохраняют контекст. Два эпизода можно чередовать в одном процессе без смешивания RNG; никакой async-код не держит глобальный current-runtime между await.

В `shared/utils/randomGenerator.js` сохранить сигнатуры `getRandom()` и inclusive `getIntRandom(min,max)`, но при активном runtime брать его RNG; без runtime поведение web и используемый тестами `replaceGetRandom` сохраняются. `replaceGetRandom` остаётся тестовым override, не механизмом среды.

Мигрировать gameplay UUID/time/random на эти функции:
- `GameModel.js`: QuestionRecord.new, GameModel.new, toDatabase timestamps, doShuffle;
- `GameModel.parse.js`, `CardModel.js`, `AnimalModel.js`, `TraitModel.js`, `plantarium/PlantModel.js`: ID; у CardModel удалить зависимость от `GLOBAL_BROWSER`, forcedId сохранить;
- `traitsData/lifecycle.js`: выбор Mammal; существующие `getIntRandom` в Running, mutations и `game.plantarium.js` используют тот же runtime;
- `shared/actions/game.js`, `trait.js`, `server/reducers/games-rdx-server.js`: фазы/вопросы/log timestamps; terminal `getRandom` тоже seeded.
Контрольный поиск `Math\.random|uuid\.v4|Date\.now|new Date|lodash/(shuffle|random|sample)` по gameplay-путям должен оставить только явно обозначенный web-default в utility, не игровые вызовы.

Новый `rl/node/runtime.js` реализует:
- `seed`: целое uint32; `0` валиден, отсутствующий seed в reset получает entropy от Node crypto один раз и возвращается в admin/replay metadata;
- RNG Mulberry32 с одним uint32 состоянием: `state += 0x6D2B79F5; t=imul(t^(t>>>15),t|1); t^=t+imul(t^(t>>>7),t|61); result=((t^(t>>>14))>>>0)/4294967296`;
- ID: независимый монотонный счётчик, строка `${kind}_${counter}`, не расходует RNG;
- логическое время начинается с 0 и увеличивается на 1 после завершённой внешней команды, не после выбора префикса;
- snapshot включает RNG state, ID counter и clock. Seed и состояние генератора никогда не входят в policy observation/info.

### 3. Создать isolated store и остановки на решениях
Новые `rl/node/environment.js`, `store.js`, `scheduler.js`, `decisions.js`. Node API:
```js
createEnvironment(config)
env.reset({seed, scenario = null})       // -> Frame
env.observe(agent)                      // -> policy observation, no mutation
env.step({decisionId, choiceIndex})     // -> Frame
env.snapshot()                         // -> opaque same-env handle
env.restore(handle)                    // -> Frame with fresh decisionId
env.exportReplay()                     // -> admin artifact
env.close()                            // idempotent
```

`config` имеет точные поля: `numPlayers=2` (2..8), `addons=[]`, `halfDeck=false`, `maxCards=null`, `randomPlayers=false`, `maxDecisions=20000`. Addon names: `base2,timeToFly,continents,bonus,plantarium,customff,lifecycle`, соответствующие существующим `addon_*`. Неизвестные ключи/дополнения — `INVALID_CONFIG`; `maxCards` только положительное целое или null. В env нет lobby pause, disconnect, real-time timeout recovery: это не стратегические действия игры.

Store: Immutable root `{games:Map(), rooms:Map()}`. Комнаты — реальные episode-seat данные `{id, users:List(playerIds), spectators:List(), settings:SettingsRecord, gameId:null}`, без auth/соединений. Реализовать root reducer только из существующего `games-rdx-server.js.reducer` плюс собственный `rooms` reducer без web `rooms-rdx-server`: `gameCreateNotify` в RL не обрабатывается, `room.gameId` остаётся `null`, иначе `checkRoomIsNotInGame` (`shared/actions/rooms.checks.js:76`) отвергнет второй reset в том же процессе. Reset выполняется в порядке: снять scheduler callbacks → отменить pending вопрос/таймеры → `gameDestroy` предыдущего gameId → заменить room записью с `gameId:null` и новыми seats → пересоздать RNG/ID/clock/boundaryEpoch/replay history → `server$…

Normal reset строит `room.settings = new SettingsRecord({maxPlayers: config.numPlayers, halfDeck, maxCards, randomPlayers, seed: null, timeTurn: SETTINGS_DEFAULT_TIME_TURN, timeTraitResponse: SETTINGS_DEFAULT_TIME_TRAIT, timeAmbush: SETTINGS_AMBUSH_TIME})` и ставит `addon_<name>: true` ровно для имён из `config.addons`; без этого `checkRoomMaxSize` (`shared/actions/rooms.checks.js:70`) отвергнет N=5..8 при default `maxPlayers: 4`. Таймеры остаются положительными default (120000/60000/10000 ms) как logical markers: отсутствие turn marker меняет `server$playerActed`. `scenario` — отдельная YAML fixture, не RNG seed. Перед `parseFromRoom` разобрать YAML тем же `yaml-js`, соединить `scenario.settings` с `SettingsRecord` defaults и constructor config; при явном к…

Scheduler реализует существующие control action types add/cancel/check; хранит `{name,remaining,callback,order}` без native `setTimeout`. `checkTimeout` видит действующий turn marker. Duration 0 сохраняет существующее «не ставить», не «выполнить сейчас». Никакие turn/question deadlines сами не истекают от скорости обучения. Время advance — только явное завершение автономной фазы, не wall clock.

`advanceUntilDecision()` сначала вычисляет primary decision по порядку ниже, затем оборачивает FEEDING normal/question решения ещё не пройденным transient window этой epoch; после окон публикует primary actor. Порядок primary:
1. FINAL → terminal frame.
2. Существующий `game.question` → actor `question.userId`; ответ DEFENSE/INTELLECT, независимо от currentPlayer. Синхронно автоматически выбранную единственную защиту не превращать в искусственный policy prompt.
3. AMBUSH preparation → собрать приватные ответы всех владельцев eligible ambushers по seat order; выборы не показывать другим. Хендлер сам завершает подготовку, когда все `ambushers !== null`. Пока идёт attack resolution, только выполнять существующие синхронные переходы, останавливаясь на возникшем question. Не истекать ambush timer вместо ещё не опрошенного игрока.
4. REGENERATION → по seat order дать владельцам flagged animals с альтернативами выбрать карту либо `USE_DEFAULT`; после отказа владельца пометить его done в env. Когда выборов не осталось, вызвать сохранённый phase-end callback `Phase#Regeneration#...` один раз: он применяет existing first-hand/deck defaults. Автоматический `server$gamePhaseCheckEndRegeneration` может завершить фазу раньше — тогда сбросить очередь.
5. DEPLOY/FEEDING → обычный currentPlayer и полный domain действий. PREPARE/EXTINCTION выполняются существующими синхронными thunks, сами не требуют policy action.
6. Неизвестная question type, пустой domain у нетерминального decision, или отсутствие перехода/решения → `ENGINE_STALLED`, не произвольный pass.

[…84ln elided…]

Из уже прочитанного кода два конкретных дефекта могут блокировать полный sweep, и их исправить в authoritative правилах с regression:
- `game.plantarium.js:70` inclusive `getIntRandom(0,plantsList.size)` может взять undefined; для непустого списка верхняя граница `size-1`; при пустом списке pests не имеют plant-target и map пуст, без RNG draw.
- `trait.js:904` default Intellect использует необъявленный `defenseAnimal`; передать уже найденный `targetAnimal`. Проверить только существующие DEFENSE/INTELLECT timer defaults, не оживлять unused PLANT_COUNTERATTACK API.
Иные обнаруженные при rollout consumer-visible rule crashes чинить в общем движке по воспроизводимому сценарию; не исключать карту/аддон из тренировочного профиля и не catch-and-pass.

Baseline policy `random`: независимо выбирать равномерно среди текущих prefix choices через отдельный policy RNG. Это намеренно **не** равномерный sampling полных compound actions; не обещать такой baseline и не вычислять весь Cartesian product ради него. `survival`: visible-only deterministic greedy. DEPLOY: если нет животных — создать из первого card в position 0; иначе поставить доступную own-animal trait с минимальной дополнительной food demand, tie-break по card/side/animal/link local order; если нет такой trait — pass. FEEDING: доступный корм own animal с минимальным положительным visible remaining need, затем доступный TraitFatTissue convert, иначе pass. DEFENSE: первый доступный не-DECLINE вариант по helper/target local order, при отсутствии — DECLIN…

`rollout.py` запускает random/survival/shared-policy self-play для 2–8 мест с собственным policy RNG; `--workers N` создаёт N независимых env subprocess и собирает циклы через thread pool, без навязывания PettingZoo Parallel API игре. CLI `--profile base` = addons=[], `--profile all` = все семь flags, `--policy random|survival` (default random), `--seed K` задаёт последовательные seeds K..K+episodes-1. `--matrix full --seeds A:B` означает включительный диапазон и по одному эпизоду на config/player-count/seed; выполнить обе baseline policies, а в отчёте разделить результаты. Padding только в batch-collator до максимального фактического размера batch, все masked logits для padding = -inf; никаких entity/action caps.

Короткое обучение: `train_smoke.py`, CPU PyTorch REINFORCE с value baseline, shared weights всех мест, gamma=1, Adam lr=3e-4, 32 завершённых base2p episodes, batch 8, seed 7, hidden size 64, entropy coefficient 0.01, clip_grad_norm 1.0. `base2p` означает базовую игру **без** addon `base2` (двойной колоды). Для type/enum features embeddings 16, counters преобразовать `x/(1+abs(x))`, отсутствующие/padded embeddings нулевые. Для каждого typed row encoder `Linear→Tanh→Linear→Tanh`, hidden/output 64; traits mean-pool внутрь entity, entities mean-pool по own/opponent/plant группам. Candidate scorer — такой же MLP над concat(global,group means,choice encoding,gathered target/source embeddings), выход один logit на choice. Value head — тот же MLP context→scalar. Для…

После обучения сохранить checkpoint `{model,optimizer,catalog,schema_version,config,seed}` и заново загрузить его для 8 evaluation episodes с фиксированными seeds 10000..10007. Требование — реальные legal rollouts, finite loss/градиенты, ненулевой gradient на scripted winning fixture, изменившийся checkpoint hash и воспроизводимый inference после reload; улучшение win rate за 32 эпизода не обещать. Если все 32 random returns случайно одинаковы, gradient-check fixture остаётся отдельным доказательством обучающего сигнала, не подменяет natural games.

Scripted gradient fixture использует существующий YAML parser:
```yaml
phase: feeding
food: 0
deck: ''
players:
  - continent: $A TraitCarnivorous, $D TraitSwimming +
  - continent: $B TraitPoisonous +, $C +, $E TraitSwimming +
```
На первом решении выбрать `A→C` либо `A→B` через обычный prefix API и доиграть existing forced transitions. Ожидается: безопасная добыча C даёт player_0 счёт 7 против 6 и reward +1; poisonous B убивает A в extinction, счёт player_0 3 против 5 и reward -1. Ни tie-break RNG, ни scripted reward не нужны. Проверка должна сначала подтвердить эти natural outcomes, затем один gradient update увеличивает log-prob безопасной ветви при фиксированном observation/initial weights.

### 9. Порядок интеграции и разделение исполнителей
Волна A: один владелец shared extraction/runtime (шаги 1–2), отдельный исполнитель Python skeleton/uv **после фиксации схем этого плана** может работать независимо, не трогая shared. Волна B: после A runtime-owner делает store/scheduler/snapshots, action-owner — validation/prefix domains, observation-owner — projection/catalog; общий integration owner `environment.js`, shared/action handler edits только у action-owner. Волна C: AEC/bridge и сценарии аддонов; затем rollout/training smoke/performance. Промежуточный base2p gate не является сдачей всего проекта.

Ни один исполнитель не запускает build/lint/full suites в середине общей волны. После интеграции каждый этап проходит связанные behavioral checks ниже; общий suite — один раз на интегрированном дереве. Не менять package-lock/toolchain целиком ради RL.

## Critical files & anchors
1. `shared/actions/game.js` — `server$playerActed`, `server$addTurnTimeout`, regeneration: отсутствие timeout меняет правила; не отключать marker.
2. `shared/actions/trait.js` — auto-sharing до проверок, вопросы и ambush prepare: stop boundaries и atomicity не равны одному Redux action.
3. `shared/actions/trait.checks.js` — ordered TWO_TRAITS и отдельные plant attack/counterattack checks; generic animal target перебор недостаточен.
4. `shared/models/game/evolution/traitsData/hunt.js` — auto-defense, intellect, Anglerfish и вложенные hunts: не отдавать policy уже автоматически решённые ветки.
5. `server/socket.js:57-89` — `meta.clientOnly` не проходит authoritative reducer; этот фильтр обязателен и без сокета.

## Verification
### Prerequisites и команды
Все команды из корня репозитория. В планировании они **не выполнялись**. Исторический pass-only smoke (111 end-turn, FINAL, deck 0) доказывает только работу test harness, не полноту RL-среды и не производительность. Зависимости JS установлены ранее; при чистом checkout использовать существующий package-lock и `npm ci --ignore-scripts`. Node24 для env, Python3.11 через uv; existing shared suite может использовать Node16 из `.nvmrc`, но navigator fix позволяет проверить и Node24.

Добавить npm scripts:
- `rl:smoke`: `node rl/node/worker.cjs --smoke`
- `rl:test`: `mocha --require @babel/register --require rl/node/test/setup.cjs "rl/node/**/*.spec.js"`
`setup.cjs` создаёт только Chai assertions/runtime fixtures; не импортирует globals/test-helper, не требует JWT. Worker launcher в `--smoke` печатает один admin JSON результат в stdout и выходит 0 только после natural terminal и checks, а не запускает JSONL loop.

```bash
npm run rl:smoke
npm run rl:test
TEST=true NODE_ENV=test JWT_SECRET=secret ./node_modules/.bin/mocha --require @babel/register --require @babel/polyfill --require source-map-support/register --require ignore-styles --require ./globals --require ./shared/test-helper.js "shared/**/*.spec.js"
uv lock --check
uv sync --locked --extra train
uv run --locked --extra train pytest rl/python/tests -q
uv run --locked python -m evolution_rl.rollout --profile base --players 2 --episodes 100 --seed 0 --workers 1
uv run --locked python -m evolution_rl.rollout --matrix full --seeds 0:19 --workers 4
uv run --locked --extra train python -m evolution_rl.train_smoke --episodes 32 --seed 7 --output rl/runs/smoke
uv run --locked python -m evolution_rl.benchmark --profile all --players 8 --episodes 20 --workers 1,4
```
При создании lockfile первый раз выполнить `uv lock`, затем все перечисленные проверки с `--locked`; dev dependency pytest включена uv по default. Запуски без extra train не импортируют torch.

### Behavioral gates
1. **Pure import/end-to-end:** при пустых MONGO_URL/JWT_SECRET, без globals/window/navigator/jsdom и с перехватом попыток native sockets/timers импортировать env, reset seed=0, разыграть минимум 2 animals, обе стороны dual card, linked trait, выполнить feeding/hunt и закончить base game в FINAL. Assert реальные animal/food/score transitions, не только `no throw`; assert require cache не содержит Mongo, socket.io, jsdom, client/actions/app. Существующий web lifecycle database-spec теперь подключает новый middleware и по-прежнему проверяет сохранение stats.
2. **Determinism/isolation:** два env seed=123 + одинаковые semantic actions → одинаковые authoritative hashes, dice, snapshots, winner; interleave с третьим seed=456 не меняет траекторию первых. Повтор `observe`, генерация choices и catalog не расходуют RNG/ID/clock. ID уникальны при ≥100000 generation; не усекать до 4 символов. Seed 0 не превращается в отсутствующий seed.
3. **Decision coverage:** scenario DEPLOY с dual-sided card, enemy parasite и linked endpoints — все стороны/позиции/targets; FEEDING с двумя животными × двумя доступными plant targets, где действие еды не должно скрыть trait actions; рекомбинация содержит все ordered pairs. Проверять admissible leaves против независимого brute-force малых сценариев + pure validator, не сравнивать enumerator с самим собой. Отдельно feeding с доступными non-eating actions сохраняет actor при marker, вопрос передаёт управление владельцу prey и возвращает правильно.
4. **Atomic errors:** malformed/stale/wrong-owner/out-of-range команда оставляет snapshot state/RNG/clock/scheduler прежним, включая auto-sharing; missing/unknown enum/payload не становится pass. Для admitted `attempt_rejected` неизменны gameplay state/RNG/scheduler, но обновлён attempt count/token. Сценарий pending sharing, после которого прежняя target насыщается, не должен менять порядок web правил и не оставляет половинчатый env transition при отказе. Публично неверный leaf без pending preparation — ENGINE_INVARIANT.
5. **Privacy:** paired states отличаются только чужими hands/deck order/types/hidden transient values/UUID/settings.seed/чужими незавершёнными ambush answers → для данного игрока obs, choices, mask byte-identical. Исключения только уже раскрытые public outcomes или его own private hand/response. Own private changes доступны только владельцу. Inherent PlantCarnivorous capability из одинакового публичного type не считается утечкой. В snapshot/replay admin скрытые данные допустимы, но они не попадают в policy info.
6. **Scheduler/snapshot:** несколько ambushers разных игроков принимают приватные решения, атаки идут прежним ordered sequence, queued food не теряется; DEFENSE→INTELLECT→FEEDING и regeneration с 2 вариантами hand не исполняются timeout-автоматически. Snapshot перед question/ambush/regen, branch, restore, повтор действий → исходные state hashes и outcome, кроме transport decision token. Callback после restore не запускается дважды; чужой handle отвергается. Повторный `reset` в том же env проходит без `checkRoomIsNotInGame`, предыдущий game удалён из `games`, `room.gameId` остаётся null; 100 reset/close в одном процессе не растят `games`. Сценарий Anglerfish + Communication/Cooperation: если auto-sharing прерывает toggle-вектор, остаток применён после разреше…
7. **Rules matrix:** все 128 комбинаций семи addon flags × каждый N=2..8, seeds 0..19, обе baseline policies (17920 эпизодов на policy, 35840 всего), плюс halfDeck/maxCards edge cases и randomPlayers. Проверять invalid selections=0 для публично детерминированных leaves, no engine errors, deck/food nonnegative, unique entity IDs, валидные surviving links, terminal consistency. Truncation из-за циклической стратегии честно записать, не засчитывать как natural terminal; config/seed должен иметь natural completion хотя бы survival baseline. Аддон-specific fixtures дополнительно покрывают: ambush/intellect/shell/metamorphose/viviparous; neoplasm/recombination/regeneration; Rstrategy; parasite covers/carnivorous plant/Honey; mutation/oviparous; spores/cyst/mammal.…
8. **Python/API/process:** `pettingzoo.test.api_test(raw_env(), num_cycles=1000)`, `seed_test(factory,num_cycles=100)`; динамические spaces повторно возвращают тот же объект в одном decision, `space.contains(obs)` для каждого агента включая inactive/terminal, маска int8 нужной длины. Все terminal rewards выдаются ровно раз через cumulative_rewards, dead step только None. Проверить malformed JSON/EOF/stale id, child crash, repeated close, отсутствие zombie processes после 100 reset/close.
9. **Learning proof:** команда train_smoke завершает 32 реальные эпизода, обновляет веса (finite gradients/loss), сохраняет и загружает checkpoint, выполняет evaluation; scripted winning branch получает положительный policy gradient против альтернативы, не mock reward. Evaluation seeds не совпадают с training seeds. Никакого требования роста win rate за короткий smoke.
10. **Throughput:** отдельно cold startup, reset median/p95, committed actions/sec и prefix choices/sec, RSS/env, candidate-domain cost, JSON bytes/step для base2p и all8p; сравнить 1/4 workers. Не выдавать скорость pass-only партии за rule-step benchmark. Оптимизировать только измеренное: кеш по state/prefix и передача actor-only view допустимы; не менять правила, не сужать action domains и не вводить лимиты сущностей ради цифры.

## Assumptions & contingencies
- Пользователь выбрал все существующие дополнения/2–8 игроков и «среда + проверка обучаемости». Изменение этого объёма требует нового решения, не автоматического отказа от сложных правил.
- uv обязателен: если Python3.11 отсутствует, `uv sync` получает managed interpreter; если uv отсутствует, установить uv стандартным способом до выполнения, не заменять pip-workflow.
- Стандартный runtime среды Node24; при несовместимости нового headless import с Node24 исправить конкретную runtime-зависимость без глобального обновления legacy frontend. Existing web Node16 профиль сохраняется.
- Отсутствие universal entity cap принято как причина Sequence/prefix модели; при слишком длинном эпизоде срабатывает только явный maxDecisions/truncation, не обрезание наблюдений/действий.
- Никаких GPU/Hugging Face/cloud обязательств; learning smoke на CPU. Если torch build недоступен для конкретной платформы, запускать указанный CPU smoke в Linux x64 Python3.11, а не подменять обучение mock policy.
- Context7 resolver дважды вернул `fetch failed`; библиотечные контракты сверены с первичными источниками: [AEC](https://pettingzoo.farama.org/api/aec/), [создание](https://pettingzoo.farama.org/content/environment_creation/), [тесты](https://pettingzoo.farama.org/content/environment_tests/), [Gym Sequence](https://gymnasium.farama.org/api/spaces/composite/), [PyTorch REINFORCE/Categorical](https://docs.pytorch.org/docs/2.14/distributions.html), [uv lock/sync](https://docs.astral.sh/uv/concepts/projects/sync/). При расхождении resolved versions зафиксировать совместимые версии внутри указанных bounds и прогнать те же acceptance checks; не менять API среды молча.


[Showing lines 1-95 and 180-259 of 259; 84 middle lines (22.2KB) elided. Read artifact://921 for full output. Some lines truncated to 768 chars. Use local://evolution-rl-plan.md:raw:82-82 to read line 82 whole; cut lines run to line 247, each read the same way]
