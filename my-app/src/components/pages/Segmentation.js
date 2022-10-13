import React from 'react';
import '../../App.css';

import { Button } from '.././Button';
import '.././HeroSection.css';

function Segmentation() {
  return (
    <div className='hero-container'>
      <video src='/videos/video-2.mp4' autoPlay loop muted />
      <div className='hero-btns'>
        <Button
          className='btns'
          buttonStyle='btn--primary'
          buttonSize='btn--large'
          onClick={(e) => {
            e.preventDefault();
            window.location.href='http://google.com';
            }}
        >
          DEMO <i className='far fa-play-circle' />
        </Button>
      </div>
    </div>
  );
}

export default Segmentation;