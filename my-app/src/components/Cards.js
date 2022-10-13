import React from 'react';
import './Cards.css';
import CardItem from './CardItem';

function Cards() {
  return (
    <div className='cards'>
      { <div className='cards__container'>
        <div className='cards__wrapper'>
          <ul className='cards__items'>
            <CardItem
              src='images/safety.jpg'
              text='Created tu assure food safety in the world with efficient silos placing for better storing ressources'
              path='/services'
            />
            <CardItem
              src='images/datascientist.jpg'
              text='A specialized team of data scientists from Ecole Polytechnique in partnership with QuantumBlack'
              path='/services'
            />
          </ul>
        </div>
      </div> }
    </div>
  );
}

export default Cards;
