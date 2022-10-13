import React from 'react';
import '../../App.css';

import { Button } from '.././Button';
import '.././HeroSection.css';

function Classification() {
  return (
    <div className='hero-container'>
      <video src='/videos/video-2.mp4' autoPlay loop muted />
      <h1>Test our product!</h1>
      <div className='hero-btns'>
        <Button
          className='btns'
          buttonStyle='btn--primary'
          buttonSize='btn--large'
          onClick={(e) => {
            e.preventDefault();
            /* here it's mandatory to have an account and a functional streamlit site working!*/
            window.location.href='https://maumlima-streamlit-hackathon-streamlit-app-n8xth1.streamlitapp.com/';
            }}
        >
          DEMO <i className='far fa-play-circle' />
        </Button>
      </div>
    </div>
  );
}

export default Classification;
