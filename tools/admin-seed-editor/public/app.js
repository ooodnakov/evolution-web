const {useState, useEffect, useMemo, useCallback, useRef} = React;

const FALLBACK_LANGUAGE = 'en';
const DEFAULT_LANGUAGE_OPTIONS = [
  {code: 'en', label: 'English'},
  {code: 'ru', label: 'Русский'}
];

const UI_TRANSLATIONS = {
  en: {
    language: {
      label: 'Language'
    },
    cards: {
      plantBadge: 'Plant'
    },
    app: {
      title: 'Evolution Admin Seed Editor',
      description: 'Build complex room setups visually and export them into the admin command text field. Drag cards into the deck or directly into player hands, tweak room options, and export a ready-to-use YAML seed.'
    },
    import: {
      title: 'Import existing seed',
      placeholder: 'Paste seed YAML here and click Import',
      loadDefault: 'Load default example',
      button: 'Import'
    },
    export: {
      title: 'Exported seed',
      placeholder: 'Click Export to generate seed',
      button: 'Export seed'
    },
    status: {
      generated: 'Seed generated.',
      copied: 'Seed copied to clipboard.',
      copyManual: 'Seed ready. Copy it manually if needed.',
      importSuccess: 'Seed imported successfully.',
      importFailed: 'Failed to parse provided seed.'
    },
    room: {
      title: 'Room configuration',
      description: 'Adjust game settings that are typically edited via the in-room admin panel.',
      phaseLabel: 'Phase',
      defaultPhase: 'Default',
      foodLabel: 'Food pool',
      nameLabel: 'Room name',
      maxPlayersLabel: 'Max players',
      turnTimeLabel: 'Turn time (seconds)',
      traitResponseLabel: 'Trait response time (seconds)',
      passwordLabel: 'Password',
      plantariumToggle: 'Enable Plantarium addon'
    },
    deck: {
      title: 'Deck configuration',
      description: 'Build the main deck that will be used to distribute cards.',
      mainTitle: 'Main deck',
      mainDescription: 'Cards will be exported as a YAML deck entry',
      mainPlaceholder: 'Drag cards from the library to add them to the deck.',
      plantTitle: 'Plant deck',
      plantDescription: 'Only exported when Plantarium addon is enabled',
      plantPlaceholder: 'Drag plant cards here.'
    },
    players: {
      title: 'Players',
      description: 'Drop cards into each player’s hand and describe their continent setup.',
      addButton: '+ Add player',
      displayLabel: 'Display label',
      removeButton: 'Remove player',
      handTitle: 'Starting hand',
      handDescription: 'Hand for {{name}}',
      handPlaceholder: 'Drag cards here to give them to the player.',
      continentLabel: 'Continent configuration (traits, animals, links)',
      continentPlaceholder: '$A TraitCarnivorous, $B TraitSymbiosis$A',
      defaultName: 'Player {{index}}'
    },
    customSettings: {
      title: 'Advanced room settings',
      description: 'Add custom key-value pairs for rare options. Values are exported exactly as entered.',
      addButton: '+ Add custom setting',
      empty: 'No custom settings. Click “Add custom setting” to append one.',
      keyLabel: 'Key',
      valueLabel: 'Value',
      removeButton: 'Remove'
    },
    library: {
      cardTitle: 'Card library',
      cardDescription: 'Drag cards into the deck or player hands.',
      plantTitle: 'Plant library',
      plantDescription: 'Plant cards available when the Plantarium addon is enabled.',
      searchPlaceholder: 'Search cards',
      count: '{{count}} cards'
    },
    phases: {
      prepare: 'Prepare',
      deploy: 'Deploy',
      feeding: 'Feeding',
      ambush: 'Ambush',
      extinction: 'Extinction',
      regeneration: 'Regeneration',
      final: 'Final'
    }
  },
  ru: {
    language: {
      label: 'Язык'
    },
    cards: {
      plantBadge: 'Растение'
    },
    app: {
      title: 'Редактор сидов администратора Evolution',
      description: 'Создавайте сложные настройки комнат в наглядном интерфейсе и экспортируйте их в поле административной команды. Перетаскивайте карты в колоду или прямо в руки игроков, настраивайте параметры комнаты и выгружайте готовый YAML-сид.'
    },
    import: {
      title: 'Импорт существующего сида',
      placeholder: 'Вставьте YAML сид сюда и нажмите «Импорт»',
      loadDefault: 'Загрузить пример по умолчанию',
      button: 'Импортировать'
    },
    export: {
      title: 'Экспортированный сид',
      placeholder: 'Нажмите «Экспорт», чтобы получить сид',
      button: 'Экспортировать сид'
    },
    status: {
      generated: 'Сид создан.',
      copied: 'Сид скопирован в буфер обмена.',
      copyManual: 'Сид готов. Скопируйте его вручную при необходимости.',
      importSuccess: 'Сид успешно импортирован.',
      importFailed: 'Не удалось разобрать указанный сид.'
    },
    room: {
      title: 'Настройки комнаты',
      description: 'Настройте параметры игры, которые обычно меняются в админ-панели комнаты.',
      phaseLabel: 'Фаза',
      defaultPhase: 'По умолчанию',
      foodLabel: 'Бассейн еды',
      nameLabel: 'Название комнаты',
      maxPlayersLabel: 'Максимум игроков',
      turnTimeLabel: 'Время хода (секунды)',
      traitResponseLabel: 'Время реакции свойства (секунды)',
      passwordLabel: 'Пароль',
      plantariumToggle: 'Включить дополнение «Плантариум»'
    },
    deck: {
      title: 'Настройка колоды',
      description: 'Соберите основную колоду, из которой будут раздавать карты.',
      mainTitle: 'Основная колода',
      mainDescription: 'Карты попадут в секцию deck в YAML',
      mainPlaceholder: 'Перетащите карты из библиотеки, чтобы добавить их в колоду.',
      plantTitle: 'Колода растений',
      plantDescription: 'Экспортируется только при включённом дополнении «Плантариум».',
      plantPlaceholder: 'Перетащите сюда карты растений.'
    },
    players: {
      title: 'Игроки',
      description: 'Раздайте карты в руки игроков и опишите их континенты.',
      addButton: '+ Добавить игрока',
      displayLabel: 'Отображаемая подпись',
      removeButton: 'Удалить игрока',
      handTitle: 'Стартовая рука',
      handDescription: 'Рука игрока {{name}}',
      handPlaceholder: 'Перетащите сюда карты, чтобы выдать их игроку.',
      continentLabel: 'Конфигурация континента (свойства, животные, связи)',
      continentPlaceholder: '$A TraitCarnivorous, $B TraitSymbiosis$A',
      defaultName: 'Игрок {{index}}'
    },
    customSettings: {
      title: 'Расширенные настройки комнаты',
      description: 'Добавьте редко используемые пары ключ-значение. Значения экспортируются без изменений.',
      addButton: '+ Добавить настройку',
      empty: 'Нет пользовательских настроек. Нажмите «Добавить настройку», чтобы создать запись.',
      keyLabel: 'Ключ',
      valueLabel: 'Значение',
      removeButton: 'Удалить'
    },
    library: {
      cardTitle: 'Библиотека карт',
      cardDescription: 'Перетаскивайте карты в колоду или руки игроков.',
      plantTitle: 'Библиотека растений',
      plantDescription: 'Карты растений доступны при включённом дополнении «Плантариум».',
      searchPlaceholder: 'Поиск карт',
      count: '{{count}} карт'
    },
    phases: {
      prepare: 'Подготовка',
      deploy: 'Размещение',
      feeding: 'Кормёжка',
      ambush: 'Засада',
      extinction: 'Вымирание',
      regeneration: 'Регенерация',
      final: 'Финал'
    }
  }
};

const createId = () => Math.random().toString(36).slice(2, 10);

const defaultSeedTemplate = `deck: 12 carnivorous, 6 sharp
phase: feeding
food: 2
players:
  - hand: 1 sharp, 1 camo
    continent: carn sharp, carn camo
  - hand: 1 sharp, 1 camo
    continent: carn sharp, carn camo
`;

const defaultSettings = () => ({
  name: '',
  maxPlayers: 4,
  timeTurn: 60,
  timeTraitResponse: 30,
  addon_plantarium: false,
  password: ''
});

const createPlayer = (index = 0, overrides = {}) => {
  const {name, hand, continent, generatedNameIndex, ...rest} = overrides;
  const resolvedName = typeof name === 'string' ? name : `Player ${index + 1}`;
  const resolvedHand = Array.isArray(hand) ? hand : [];
  const resolvedContinent = typeof continent === 'string' ? continent : '';
  const hasGeneratedOverride = Object.prototype.hasOwnProperty.call(overrides, 'generatedNameIndex');
  return {
    id: createId(),
    name: resolvedName,
    hand: resolvedHand,
    continent: resolvedContinent,
    generatedNameIndex: hasGeneratedOverride ? generatedNameIndex : (typeof name === 'string' ? null : index),
    ...rest
  };
};

const createCardToken = (meta) => {
  const seedName = (meta && meta.seedName) || (typeof meta === 'string' ? meta : 'card');
  return {
    uid: createId(),
    seedName,
    name: (meta && meta.name) || (meta && meta.seedName) || (typeof meta === 'string' ? meta : 'Card'),
    traits: meta && Array.isArray(meta.traits) ? [...meta.traits] : [],
    category: (meta && meta.category) || 'card'
  };
};

const useSortable = (ref, options) => {
  useEffect(() => {
    if (!ref.current) return;
    const sortable = new Sortable(ref.current, options);
    return () => sortable.destroy();
  }, [ref, options]);
};

const getListRef = (draft, listId) => {
  if (!listId) return null;
  if (listId === 'deck') return draft.deck;
  if (listId === 'deckPlants') return draft.deckPlants;
  const match = listId.match(/^player-(.+?)-hand$/);
  if (match) {
    const player = draft.players.find((p) => p.id === match[1]);
    return player ? player.hand : null;
  }
  return null;
};

const formatSettingValue = (value) => {
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') return value;
  return `${value}`;
};

const buildCardString = (cards, lookup) => {
  if (!cards || cards.length === 0) return '';
  const counter = cards.reduce((acc, card) => {
    const key = card.seedName || card.name.toLowerCase();
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
  return Object.entries(counter)
    .map(([slug, count]) => {
      const meta = lookup.get(slug) || {seedName: slug};
      const token = meta.seedName || slug;
      return count > 1 ? `${count} ${token}` : `${token}`;
    })
    .join(', ');
};

const resolveFromLibrary = (slug, libraryList) => {
  if (!slug) return null;
  const normalized = slug.trim().toLowerCase();
  return libraryList.find((item) => item.seedName === normalized || item.search.includes(normalized)) || null;
};

const expandCardList = (raw, libraryList, category = 'card') => {
  if (!raw) return [];
  let items = [];
  if (Array.isArray(raw)) {
    items = raw;
  } else if (typeof raw === 'string') {
    items = raw.split(',').map((token) => token.trim()).filter(Boolean);
  } else {
    return [];
  }

  const expanded = [];
  items.forEach((token) => {
    if (!token) return;
    const parts = token.trim().split(/\s+/);
    let count = parseInt(parts[0], 10);
    let slug;
    if (Number.isFinite(count)) {
      slug = parts.slice(1).join(' ');
    } else {
      count = 1;
      slug = token.trim();
    }
    const metaFromLibrary = resolveFromLibrary(slug, libraryList);
    const meta = metaFromLibrary ? {...metaFromLibrary} : {seedName: slug, name: slug, category};
    if (!meta.category) meta.category = category;
    for (let i = 0; i < count; i += 1) {
      expanded.push(createCardToken(meta));
    }
  });
  return expanded;
};

const parseSeedString = (seedString, library, options = {}) => {
  if (!seedString) return null;
  let parsed;
  try {
    parsed = window.jsyaml.load(seedString) || {};
  } catch (err) {
    console.error('Failed to parse seed', err);
    return null;
  }
  const createPlayerName = options.createPlayerName || ((index) => `Player ${index + 1}`);
  const baseSettings = defaultSettings();
  const incomingSettings = parsed.settings || {};
  const customSettings = [];
  Object.entries(incomingSettings).forEach(([key, value]) => {
    if (Object.prototype.hasOwnProperty.call(baseSettings, key)) {
      baseSettings[key] = value;
    } else {
      customSettings.push({id: createId(), key, value: value});
    }
  });

  const players = (Array.isArray(parsed.players) && parsed.players.length > 0)
    ? parsed.players.map((player, index) => {
        const hasName = player && typeof player.name === 'string' && player.name.trim().length > 0;
        const resolvedName = hasName ? player.name : createPlayerName(index);
        return createPlayer(index, {
          name: resolvedName,
          generatedNameIndex: hasName ? null : index,
          hand: expandCardList(player && player.hand, library.cards, 'card'),
          continent: player && player.continent ? player.continent : ''
        });
      })
    : [createPlayer(0, {name: createPlayerName(0), generatedNameIndex: 0})];

  return {
    phase: parsed.phase ? String(parsed.phase).toLowerCase() : '',
    food: typeof parsed.food === 'number' ? parsed.food : '',
    deck: expandCardList(parsed.deck, library.cards, 'card'),
    deckPlants: expandCardList(parsed.deckPlants, library.plants, 'plant'),
    players,
    settings: baseSettings,
    customSettings
  };
};

const getNestedValue = (object, path) => path.reduce((acc, segment) => {
  if (acc && typeof acc === 'object' && Object.prototype.hasOwnProperty.call(acc, segment)) {
    return acc[segment];
  }
  return null;
}, object);

const formatTemplate = (template, params = {}) => {
  if (typeof template !== 'string') return template;
  return template.replace(/{{\s*([^}]+?)\s*}}/g, (_, token) => {
    const key = token.trim();
    return Object.prototype.hasOwnProperty.call(params, key) ? params[key] : '';
  });
};

const humanizeIdentifier = (value = '') => {
  if (!value) return '';
  const cleaned = value
    .replace(/^(Card|Trait|PlantTrait|Plant)/, '')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/[_-]+/g, ' ')
    .trim();
  if (!cleaned) return value;
  return cleaned
    .split(' ')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
};

const createUiTranslator = (language) => (key, params = {}, fallbackValue = null) => {
  const dictionary = UI_TRANSLATIONS[language] || UI_TRANSLATIONS[FALLBACK_LANGUAGE] || {};
  const fallbackDictionary = UI_TRANSLATIONS[FALLBACK_LANGUAGE] || {};
  const path = key.split('.');
  const value = getNestedValue(dictionary, path);
  const fallback = getNestedValue(fallbackDictionary, path);
  const template = typeof value === 'string' ? value : (typeof fallback === 'string' ? fallback : fallbackValue);
  if (template == null) return fallbackValue != null ? fallbackValue : key;
  return formatTemplate(template, params);
};

const CardToken = ({variant, lines, badge, onRemove, dataAttributes = {}}) => {
  const attributes = {...dataAttributes};
  const className = `card-token card-token--${variant}`;
  return (
    <div className={className} {...attributes}>
      {badge ? <span className="card-token__badge">{badge}</span> : null}
      <div className="card-token__body">
        {lines.map((line, index) => (
          <div key={`${line}-${index}`} className="card-token__line">{line}</div>
        ))}
      </div>
      {onRemove ? (
        <button
          className="card-token__remove"
          onClick={(event) => {
            event.stopPropagation();
            event.preventDefault();
            onRemove();
          }}
        >
          ×
        </button>
      ) : null}
    </div>
  );
};

const CardDropZone = ({title, description, listId, items, onDrop, onRemoveCard, placeholder, formatToken, groupName = 'cards'}) => {
  const containerRef = useRef(null);
  useSortable(containerRef, useMemo(() => ({
    group: {name: groupName, pull: true, put: true},
    animation: 150,
    sort: true,
    draggable: '.card-token',
    onEnd: onDrop
  }), [onDrop, groupName]));

  return (
    <div className="drop-zone">
      <div className="section-header" style={{marginBottom: '12px'}}>
        <div>
          <h3 style={{margin: 0}}>{title}</h3>
          {description ? <p className="section-description">{description}</p> : null}
        </div>
      </div>
      <div className="card-list" ref={containerRef} data-list-id={listId}>
        {items && items.length > 0
          ? items.map((card) => {
              const visual = formatToken(card);
              return (
                <CardToken
                  key={card.uid}
                  variant={visual.variant}
                  lines={visual.lines}
                  badge={visual.badge}
                  onRemove={onRemoveCard ? () => onRemoveCard(listId, card.uid) : null}
                  dataAttributes={{'data-uid': card.uid, 'data-slug': card.seedName}}
                />
              );
            })
          : null}
      </div>
      {(!items || items.length === 0) ? <p className="drop-zone__placeholder">{placeholder}</p> : null}
    </div>
  );
};

const CardLibrary = ({title, description, cards, groupName = 'cards', formatToken, searchPlaceholder, formatCountLabel}) => {
  const [query, setQuery] = useState('');
  const containerRef = useRef(null);
  useSortable(containerRef, useMemo(() => ({
    group: {name: groupName, pull: 'clone', put: false},
    animation: 150,
    sort: false,
    draggable: '.card-token'
  }), [groupName]));

  const filteredCards = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return cards;
    return cards.filter((card) => {
      const visual = formatToken(card);
      const text = visual.lines.join(' ').toLowerCase();
      return card.search.includes(normalized) || text.includes(normalized);
    });
  }, [query, cards, formatToken]);

  const countLabel = formatCountLabel(filteredCards.length);

  return (
    <section>
      <div className="section-header">
        <div>
          <h2>{title}</h2>
          <p className="section-description">{description}</p>
        </div>
        <span className="badge">{countLabel}</span>
      </div>
      <div className="library-search">
        <input
          type="text"
          placeholder={searchPlaceholder}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </div>
      <div className="card-list" ref={containerRef} data-list-id={groupName === 'cards' ? 'library' : 'plant-library'}>
        {filteredCards.map((card) => {
          const visual = formatToken(card);
          return (
            <CardToken
              key={card.id}
              variant={visual.variant}
              lines={visual.lines}
              badge={visual.badge}
              dataAttributes={{'data-slug': card.seedName}}
            />
          );
        })}
      </div>
    </section>
  );
};

const CustomSettingsEditor = ({settings, onChange, t}) => (
  <section>
    <div className="section-header">
      <div>
        <h2>{t('customSettings.title')}</h2>
        <p className="section-description">{t('customSettings.description')}</p>
      </div>
      <button className="secondary" onClick={() => onChange('add')}>
        {t('customSettings.addButton')}
      </button>
    </div>
    {settings.length === 0 ? (
      <p className="section-description">{t('customSettings.empty')}</p>
    ) : (
      settings.map((entry) => (
        <div key={entry.id} className="grid-two-columns" style={{marginBottom: '12px'}}>
          <div>
            <label>{t('customSettings.keyLabel')}</label>
            <input
              type="text"
              value={entry.key}
              onChange={(event) => onChange('update', entry.id, {key: event.target.value})}
            />
          </div>
          <div>
            <label>{t('customSettings.valueLabel')}</label>
            <input
              type="text"
              value={entry.value}
              onChange={(event) => onChange('update', entry.id, {value: event.target.value})}
            />
          </div>
          <div style={{display: 'flex', alignItems: 'flex-end'}}>
            <button className="danger" onClick={() => onChange('remove', entry.id)}>{t('customSettings.removeButton')}</button>
          </div>
        </div>
      ))
    )}
  </section>
);

const buildSettingsEntries = (settings, customSettings) => {
  const entries = Object.entries(settings)
    .filter(([key, value]) => {
      if (typeof value === 'boolean') return value;
      if (value === '' || value === null || value === undefined) return false;
      return true;
    })
    .map(([key, value]) => [key, value]);
  customSettings.forEach((entry) => {
    if (entry.key && entry.key.trim()) {
      entries.push([entry.key.trim(), entry.value]);
    }
  });
  return entries;
};

const buildSeedString = (config, cardLookup, plantLookup) => {
  const lines = [];
  const deckString = buildCardString(config.deck, cardLookup);
  const plantDeckString = buildCardString(config.deckPlants, plantLookup);
  if (deckString) lines.push(`deck: ${deckString}`);
  if (plantDeckString) lines.push(`deckPlants: ${plantDeckString}`);
  if (config.phase) lines.push(`phase: ${config.phase}`);
  if (config.food !== '' && config.food !== null && config.food !== undefined) lines.push(`food: ${config.food}`);
  lines.push('players:');
  config.players.forEach((player) => {
    const handString = buildCardString(player.hand, cardLookup);
    const continent = player.continent ? player.continent.trim() : '';
    if (handString && continent) {
      lines.push(`  - hand: ${handString}`);
      lines.push(`    continent: ${continent}`);
    } else if (handString) {
      lines.push(`  - hand: ${handString}`);
    } else if (continent) {
      lines.push(`  - continent: ${continent}`);
    } else {
      lines.push('  - {}');
    }
  });

  const settingsEntries = buildSettingsEntries(config.settings, config.customSettings);
  if (settingsEntries.length) {
    lines.push('settings:');
    settingsEntries.forEach(([key, value]) => {
      lines.push(`  ${key}: ${formatSettingValue(value)}`);
    });
  }
  return lines.join('\n');
};

const RoomSettings = ({config, onUpdate, phases, t, getPhaseLabel}) => (
  <section>
    <div className="section-header">
      <div>
        <h2>{t('room.title')}</h2>
        <p className="section-description">{t('room.description')}</p>
      </div>
    </div>
    <div className="settings-grid">
      <div>
        <label>{t('room.phaseLabel')}</label>
        <select value={config.phase} onChange={(event) => onUpdate('phase', event.target.value)}>
          <option value="">{t('room.defaultPhase')}</option>
          {phases.map((phase) => (
            <option key={phase} value={phase}>{getPhaseLabel(phase)}</option>
          ))}
        </select>
      </div>
      <div>
        <label>{t('room.foodLabel')}</label>
        <input
          type="number"
          value={config.food}
          onChange={(event) => onUpdate('food', event.target.value === '' ? '' : Number(event.target.value))}
        />
      </div>
      <div>
        <label>{t('room.nameLabel')}</label>
        <input
          type="text"
          value={config.settings.name}
          onChange={(event) => onUpdate('settings', {name: event.target.value})}
        />
      </div>
      <div>
        <label>{t('room.maxPlayersLabel')}</label>
        <input
          type="number"
          value={config.settings.maxPlayers}
          onChange={(event) => onUpdate('settings', {maxPlayers: event.target.value === '' ? '' : Number(event.target.value)})}
        />
      </div>
      <div>
        <label>{t('room.turnTimeLabel')}</label>
        <input
          type="number"
          value={config.settings.timeTurn}
          onChange={(event) => onUpdate('settings', {timeTurn: event.target.value === '' ? '' : Number(event.target.value)})}
        />
      </div>
      <div>
        <label>{t('room.traitResponseLabel')}</label>
        <input
          type="number"
          value={config.settings.timeTraitResponse}
          onChange={(event) => onUpdate('settings', {timeTraitResponse: event.target.value === '' ? '' : Number(event.target.value)})}
        />
      </div>
      <div>
        <label>{t('room.passwordLabel')}</label>
        <input
          type="text"
          value={config.settings.password}
          onChange={(event) => onUpdate('settings', {password: event.target.value})}
        />
      </div>
      <div className="inline" style={{marginTop: '28px'}}>
        <input
          id="addon-plantarium"
          type="checkbox"
          checked={!!config.settings.addon_plantarium}
          onChange={(event) => onUpdate('settings', {addon_plantarium: event.target.checked})}
        />
        <label htmlFor="addon-plantarium">{t('room.plantariumToggle')}</label>
      </div>
    </div>
  </section>
);

const PlayersEditor = ({players, onAdd, onRemove, onUpdateName, onUpdateContinent, onRemoveCard, onDrop, formatToken, t}) => (
  <section>
    <div className="section-header">
      <div>
        <h2>{t('players.title')}</h2>
        <p className="section-description">{t('players.description')}</p>
      </div>
      <button className="secondary" onClick={onAdd}>{t('players.addButton')}</button>
    </div>
    {players.map((player, index) => {
      const displayName = player.name || t('players.defaultName', {index: index + 1});
      const handDescription = t('players.handDescription', {name: displayName, index: index + 1});
      return (
        <div key={player.id} className="player-card">
          <div className="player-header">
            <div>
              <label>{t('players.displayLabel')}</label>
              <input
                type="text"
                value={player.name}
                onChange={(event) => onUpdateName(player.id, event.target.value)}
              />
            </div>
            <button className="danger" onClick={() => onRemove(player.id)} disabled={players.length <= 1}>
              {t('players.removeButton')}
            </button>
          </div>
          <CardDropZone
            title={t('players.handTitle')}
            description={handDescription}
            listId={`player-${player.id}-hand`}
            items={player.hand}
            onDrop={onDrop}
            onRemoveCard={onRemoveCard}
            placeholder={t('players.handPlaceholder')}
            formatToken={formatToken}
            groupName="cards"
          />
          <div style={{marginTop: '16px'}}>
            <label>{t('players.continentLabel')}</label>
            <textarea
              value={player.continent}
              placeholder={t('players.continentPlaceholder')}
              onChange={(event) => onUpdateContinent(player.id, event.target.value)}
            />
          </div>
        </div>
      );
    })}
  </section>
);

function App() {
  const [library, setLibrary] = useState({
    cards: [],
    traits: [],
    plants: [],
    phases: [],
    translations: {
      languages: DEFAULT_LANGUAGE_OPTIONS,
      traitLabels: {},
      plantLabels: {}
    }
  });
  const [config, setConfig] = useState({
    phase: '',
    food: '',
    deck: [],
    deckPlants: [],
    players: [createPlayer(0)],
    settings: defaultSettings(),
    customSettings: []
  });
  const [exportedSeed, setExportedSeed] = useState('');
  const [importText, setImportText] = useState('');
  const [statusKey, setStatusKey] = useState(null);
  const [language, setLanguage] = useState(FALLBACK_LANGUAGE);
  const initializedRef = useRef(false);
  const languageInitializedRef = useRef(false);

  const mutateConfig = useCallback((mutator) => {
    setConfig((prev) => {
      const draft = {
        ...prev,
        deck: prev.deck.map((card) => ({...card})),
        deckPlants: prev.deckPlants.map((card) => ({...card})),
        players: prev.players.map((player) => ({
          ...player,
          hand: player.hand.map((card) => ({...card}))
        })),
        settings: {...prev.settings},
        customSettings: prev.customSettings.map((entry) => ({...entry}))
      };
      mutator(draft);
      return draft;
    });
  }, []);

  useEffect(() => {
    fetch('/api/library')
      .then((response) => response.json())
      .then((data) => {
        setLibrary((current) => ({
          ...current,
          ...data,
          translations: data.translations || current.translations
        }));
      })
      .catch((error) => {
        console.error('Failed to load library', error);
      });
  }, []);

  const availableLanguageOptions = useMemo(() => {
    const options = library.translations && Array.isArray(library.translations.languages)
      ? library.translations.languages
      : [];
    return options.length ? options : DEFAULT_LANGUAGE_OPTIONS;
  }, [library.translations]);

  useEffect(() => {
    if (languageInitializedRef.current) return;
    if (!availableLanguageOptions || !availableLanguageOptions.length) return;
    let initial = FALLBACK_LANGUAGE;
    try {
      const stored = typeof window !== 'undefined' ? window.localStorage.getItem('seedEditorLanguage') : null;
      if (stored && availableLanguageOptions.some((option) => option.code === stored)) {
        initial = stored;
      } else if (typeof navigator !== 'undefined' && navigator.language) {
        const normalized = navigator.language.toLowerCase();
        const matched = availableLanguageOptions.find((option) => normalized.startsWith(option.code.toLowerCase()));
        if (matched) initial = matched.code;
      }
    } catch (error) {
      // ignore storage errors
    }
    languageInitializedRef.current = true;
    setLanguage(initial);
  }, [availableLanguageOptions]);

  useEffect(() => {
    if (!languageInitializedRef.current) return;
    try {
      if (typeof window !== 'undefined') {
        window.localStorage.setItem('seedEditorLanguage', language);
      }
    } catch (error) {
      // ignore storage errors
    }
  }, [language]);

  const translator = useMemo(() => createUiTranslator(language), [language]);
  const t = useCallback((key, params = {}, fallbackValue = null) => translator(key, params, fallbackValue), [translator]);
  const formatPlayerName = useCallback((index) => t('players.defaultName', {index: index + 1}), [t]);

  useEffect(() => {
    if (!initializedRef.current && library.cards.length) {
      initializedRef.current = true;
      const parsed = parseSeedString(defaultSeedTemplate, library, {createPlayerName: formatPlayerName});
      if (parsed) {
        setConfig((current) => ({
          ...current,
          ...parsed
        }));
        const trimmed = defaultSeedTemplate.trim();
        setExportedSeed(trimmed);
        setImportText(trimmed);
      }
    }
  }, [library, formatPlayerName]);

  const cardLookup = useMemo(() => {
    const map = new Map();
    library.cards.forEach((card) => map.set(card.seedName, card));
    return map;
  }, [library.cards]);

  const plantLookup = useMemo(() => {
    const map = new Map();
    library.plants.forEach((plant) => map.set(plant.seedName, plant));
    return map;
  }, [library.plants]);

  const traitLabels = useMemo(() => (
    (library.translations && library.translations.traitLabels && library.translations.traitLabels[language]) || {}
  ), [library.translations, language]);
  const fallbackTraitLabels = useMemo(() => (
    (library.translations && library.translations.traitLabels && library.translations.traitLabels[FALLBACK_LANGUAGE]) || {}
  ), [library.translations]);
  const plantLabels = useMemo(() => (
    (library.translations && library.translations.plantLabels && library.translations.plantLabels[language]) || {}
  ), [library.translations, language]);
  const fallbackPlantLabels = useMemo(() => (
    (library.translations && library.translations.plantLabels && library.translations.plantLabels[FALLBACK_LANGUAGE]) || {}
  ), [library.translations]);

  const formatCardVisual = useCallback((token) => {
    if (!token) {
      return {variant: 'card', lines: ['']};
    }
    const variant = token.category === 'plant' ? 'plant' : 'card';
    if (variant === 'plant') {
      const label = plantLabels[token.seedName]
        || fallbackPlantLabels[token.seedName]
        || humanizeIdentifier(token.seedName);
      return {variant, lines: [label], badge: t('cards.plantBadge')};
    }
    const meta = cardLookup.get(token.seedName) || token;
    const sourceTraits = Array.isArray(token.traits) && token.traits.length
      ? token.traits
      : (meta && Array.isArray(meta.traits) ? meta.traits : []);
    const uniqueTraits = Array.from(new Set(sourceTraits));
    const traitLines = uniqueTraits.map((traitKey) => (
      traitLabels[traitKey]
      || fallbackTraitLabels[traitKey]
      || humanizeIdentifier(traitKey)
    ));
    if (traitLines.length === 0) {
      const fallbackName = (meta && meta.name) || token.name || humanizeIdentifier(token.seedName);
      traitLines.push(fallbackName);
    }
    return {variant, lines: traitLines};
  }, [cardLookup, traitLabels, fallbackTraitLabels, plantLabels, fallbackPlantLabels, t]);

  useEffect(() => {
    if (!languageInitializedRef.current) return;
    mutateConfig((draft) => {
      draft.players = draft.players.map((player) => {
        if (typeof player.generatedNameIndex === 'number') {
          return {
            ...player,
            name: formatPlayerName(player.generatedNameIndex)
          };
        }
        return player;
      });
    });
  }, [formatPlayerName, mutateConfig]);

  const handleDrop = useCallback((evt) => {
    const toListId = evt.to.dataset.listId;
    const fromListId = evt.from.dataset.listId;
    if (!toListId) return;

    const slug = evt.item.dataset.slug;
    const uid = evt.item.dataset.uid;
    const newIndex = evt.newIndex;

    const pickLookup = (listId) => (listId === 'deckPlants' || fromListId === 'deckPlants' || listId === 'plant-library') ? plantLookup : cardLookup;

    if (fromListId === 'library' || fromListId === 'plant-library') {
      const lookup = pickLookup(toListId);
      const meta = lookup.get(slug) || {seedName: slug, name: slug, category: fromListId === 'plant-library' ? 'plant' : 'card'};
      mutateConfig((draft) => {
        const list = getListRef(draft, toListId);
        if (!list) return;
        list.splice(newIndex, 0, createCardToken(meta));
      });
      if (evt.item && evt.item.parentNode) {
        evt.item.parentNode.removeChild(evt.item);
      }
      return;
    }

    mutateConfig((draft) => {
      const source = getListRef(draft, fromListId);
      const target = getListRef(draft, toListId);
      if (!source || !target) return;
      const index = source.findIndex((card) => card.uid === uid);
      if (index === -1) return;
      const [card] = source.splice(index, 1);
      target.splice(newIndex, 0, card);
    });
  }, [cardLookup, plantLookup, mutateConfig]);

  const handleRemoveCard = useCallback((listId, uid) => {
    mutateConfig((draft) => {
      const list = getListRef(draft, listId);
      if (!list) return;
      const index = list.findIndex((card) => card.uid === uid);
      if (index !== -1) list.splice(index, 1);
    });
  }, [mutateConfig]);

  const handleAddPlayer = useCallback(() => {
    mutateConfig((draft) => {
      const nextIndex = draft.players.length;
      draft.players.push(createPlayer(nextIndex, {
        name: formatPlayerName(nextIndex),
        generatedNameIndex: nextIndex
      }));
    });
  }, [mutateConfig, formatPlayerName]);

  const handleRemovePlayer = useCallback((playerId) => {
    mutateConfig((draft) => {
      if (draft.players.length <= 1) return;
      draft.players = draft.players.filter((player) => player.id !== playerId);
    });
  }, [mutateConfig]);

  const handleUpdatePlayerName = useCallback((playerId, value) => {
    mutateConfig((draft) => {
      const player = draft.players.find((p) => p.id === playerId);
      if (player) {
        player.name = value;
        if (value && value.trim().length) {
          player.generatedNameIndex = null;
        }
      }
    });
  }, [mutateConfig]);

  const handleUpdatePlayerContinent = useCallback((playerId, value) => {
    mutateConfig((draft) => {
      const player = draft.players.find((p) => p.id === playerId);
      if (player) player.continent = value;
    });
  }, [mutateConfig]);

  const handleRoomUpdate = useCallback((field, value) => {
    mutateConfig((draft) => {
      if (field === 'phase') {
        draft.phase = value;
      } else if (field === 'food') {
        draft.food = value;
      } else if (field === 'settings') {
        draft.settings = {...draft.settings, ...value};
      }
    });
  }, [mutateConfig]);

  const handleCustomSettingsChange = useCallback((action, id, payload) => {
    mutateConfig((draft) => {
      if (action === 'add') {
        draft.customSettings.push({id: createId(), key: '', value: ''});
      }
      if (action === 'remove') {
        draft.customSettings = draft.customSettings.filter((entry) => entry.id !== id);
      }
      if (action === 'update') {
        const entry = draft.customSettings.find((item) => item.id === id);
        if (entry) Object.assign(entry, payload);
      }
    });
  }, [mutateConfig]);

  const handleExport = useCallback(() => {
    const seedString = buildSeedString(config, cardLookup, plantLookup);
    setExportedSeed(seedString);
    setStatusKey('status.generated');
    if (navigator.clipboard && seedString) {
      navigator.clipboard.writeText(seedString).then(() => {
        setStatusKey('status.copied');
      }).catch(() => {
        setStatusKey('status.copyManual');
      });
    }
  }, [config, cardLookup, plantLookup]);

  const handleImport = useCallback(() => {
    if (!importText.trim()) return;
    const parsed = parseSeedString(importText, library, {createPlayerName: formatPlayerName});
    if (parsed) {
      setConfig((current) => ({
        ...current,
        ...parsed
      }));
      setStatusKey('status.importSuccess');
    } else {
      setStatusKey('status.importFailed');
    }
  }, [importText, library, formatPlayerName]);

  const getPhaseLabel = useCallback((phase) => {
    if (!phase) return '';
    const fallback = humanizeIdentifier(phase);
    return t(`phases.${phase}`, {}, fallback);
  }, [t]);

  const formatLibraryCount = useCallback((count) => t('library.count', {count}), [t]);

  const languageOptions = availableLanguageOptions;
const selectedLanguage = languageOptions.some((option) => option.code === language)
  ? language
  : languageOptions[0].code;

  const statusMessage = statusKey ? t(statusKey) : '';

  return (
    <div className="app-container">
      <div className="language-toggle">
        <label htmlFor="language-select">{t('language.label')}</label>
        <select
          id="language-select"
          value={selectedLanguage}
          onChange={(event) => setLanguage(event.target.value)}
        >
          {languageOptions.map((option) => (
            <option key={option.code} value={option.code}>{option.label}</option>
          ))}
        </select>
      </div>

      <section>
        <div className="section-header">
          <div>
            <h1>{t('app.title')}</h1>
            <p className="section-description">{t('app.description')}</p>
          </div>
        </div>
        <div className="grid-two-columns">
          <div>
            <label>{t('import.title')}</label>
            <textarea
              value={importText}
              onChange={(event) => setImportText(event.target.value)}
              placeholder={t('import.placeholder')}
              style={{minHeight: '160px'}}
            />
            <div style={{marginTop: '8px', display: 'flex', gap: '12px'}}>
              <button className="secondary" onClick={() => setImportText(defaultSeedTemplate.trim())}>{t('import.loadDefault')}</button>
              <button className="primary" onClick={handleImport}>{t('import.button')}</button>
            </div>
          </div>
          <div className="export-area">
            <label>{t('export.title')}</label>
            <textarea value={exportedSeed} readOnly placeholder={t('export.placeholder')} />
            <div style={{marginTop: '8px'}}>
              <button className="primary" onClick={handleExport}>{t('export.button')}</button>
            </div>
            {statusMessage ? <p className="status-message">{statusMessage}</p> : null}
          </div>
        </div>
      </section>

      <RoomSettings config={config} onUpdate={handleRoomUpdate} phases={library.phases || []} t={t} getPhaseLabel={getPhaseLabel} />

      <section>
        <div className="section-header">
          <div>
            <h2>{t('deck.title')}</h2>
            <p className="section-description">{t('deck.description')}</p>
          </div>
        </div>
        <CardDropZone
          title={t('deck.mainTitle')}
          description={t('deck.mainDescription')}
          listId="deck"
          items={config.deck}
          onDrop={handleDrop}
          onRemoveCard={handleRemoveCard}
          placeholder={t('deck.mainPlaceholder')}
          formatToken={formatCardVisual}
          groupName="cards"
        />
        {config.settings.addon_plantarium ? (
          <div style={{marginTop: '16px'}}>
            <CardDropZone
              title={t('deck.plantTitle')}
              description={t('deck.plantDescription')}
              listId="deckPlants"
              items={config.deckPlants}
              onDrop={handleDrop}
              onRemoveCard={handleRemoveCard}
              placeholder={t('deck.plantPlaceholder')}
              formatToken={formatCardVisual}
              groupName="plants"
            />
          </div>
        ) : null}
      </section>

      <PlayersEditor
        players={config.players}
        onAdd={handleAddPlayer}
        onRemove={handleRemovePlayer}
        onUpdateName={handleUpdatePlayerName}
        onUpdateContinent={handleUpdatePlayerContinent}
        onRemoveCard={handleRemoveCard}
        onDrop={handleDrop}
        formatToken={formatCardVisual}
        t={t}
      />

      <CustomSettingsEditor settings={config.customSettings} onChange={handleCustomSettingsChange} t={t} />

      <CardLibrary
        title={t('library.cardTitle')}
        description={t('library.cardDescription')}
        cards={library.cards}
        groupName="cards"
        formatToken={formatCardVisual}
        searchPlaceholder={t('library.searchPlaceholder')}
        formatCountLabel={formatLibraryCount}
      />
      {config.settings.addon_plantarium ? (
        <CardLibrary
          title={t('library.plantTitle')}
          description={t('library.plantDescription')}
          cards={library.plants}
          groupName="plants"
          formatToken={formatCardVisual}
          searchPlaceholder={t('library.searchPlaceholder')}
          formatCountLabel={formatLibraryCount}
        />
      ) : null}
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById('root'));
