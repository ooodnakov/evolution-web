import {Record, Range} from 'immutable';

import {getTraitDataModel} from './evolution/TraitModel'
import * as cardsData from './evolution/cards/index'
import {gameId} from '../../utils/gameRuntime'


export class CardModel extends Record({
  id: null
  , type: null
  , image: null
  , trait1: null
  , trait2: null
}) {
  static new(cardType, forcedId) {
    if (!(cardType in cardsData)) throw Error(`card#${JSON.stringify(cardType)} not found`);
    const cardData = cardsData[cardType];
    const id = forcedId // if forced, keep the caller-supplied id
      || gameId('card');
    return CardModel.fromServer({
      id
      , ...cardData
    });
  }

  static generate(count, cardType = 'CardUnknown') {
    return Range(0, count).map(i => CardModel.new(cardType)).toList();
  }

  static fromServer(js) {
    if (js !== null && js.type !== 'CardUnknown' && js.trait1 == null) {
      throw new Error(`Card ${js.type} doesn't have trait #1`)
    }
    return js == null
      ? null
      : new CardModel(js)
  }

  toClient() {
    return this
  }

  toOthers() {
    return CardModel.new(cardsData.CardUnknown.type, this.id)
  }

  get traitsCount() {
    return (this.trait1 === null ? 0
      : this.trait2 === null ? 1
      : 2)
  }

  getTraitDataModel(alternateTrait) {
    return getTraitDataModel(!alternateTrait ? this.trait1 : this.trait2);
  }

  toString() {
    return `Card #${this.id} (${this.type})`;
  }
}