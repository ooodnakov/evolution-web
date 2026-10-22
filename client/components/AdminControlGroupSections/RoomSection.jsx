import React, {Component} from 'react';
import {connect} from 'react-redux';

import {roomSetSeedRequest, roomStartVotingRequest} from '../../../shared/actions/actions';
import Button from "@material-ui/core/Button";

import SeedEditor from './SeedEditor/SeedEditor.jsx';
import {defaultSeedString} from './SeedEditor/seedUtils';

export class RoomSection extends Component {
  constructor(props) {
    super(props);
    this.state = {
      gameSeed: window.localStorage.getItem('gameSeed') || defaultSeedString
    }
  }

  setGameSeed(gameSeed) {
    window.localStorage.setItem('gameSeed', gameSeed);
    this.setState({gameSeed})
  }

  render() {
    return <div>
      {this.props.gameCanStart
        ? <Button variant='outlined'
                  fullWidth={true}
                  onClick={this.props.$start(this.props.roomId, this.state.gameSeed)}>
          Start Game
        </Button>
        : null}
      <SeedEditor
        seed={this.state.gameSeed}
        onChange={(value) => this.setGameSeed(value)}
        roomPlayerCount={this.props.roomPlayersCount}
      />
    </div>
  }
}

export default connect(
  (state) => {
    const userId = state.getIn(['user', 'id'], '%USERNAME%');
    const roomId = state.get('room');
    const room = state.getIn(['rooms', roomId]);
    const gameCanStart = room ? room.checkCanStart(userId) : false;
    const roomPlayersCount = room ? room.users.size : 0;
    return {
      roomId
      , userId
      , gameCanStart
      , roomPlayersCount
    }
  }
  , (dispatch) => ({
    $start: (roomId, seed) => () => {
      dispatch(roomSetSeedRequest(seed));
      dispatch(roomStartVotingRequest());
    }
  })
)(RoomSection);
