import React from 'react';
import '../App.css';
import { Button } from './Button';
import './HeroSection.css';

function HeroSection() {
  return (
    <div className='hero-container'>
      <video src='/videos/video1.mp4' autoPlay loop muted />
      <h1>SILOS DETECT</h1>
      <p>Solve the worldwide food problem with deep learning now!</p>
      <div className='hero-btns'>
        <Button
          className='btns'
          buttonStyle='btn--outline'
          buttonSize='btn--large'
        >
          Get in Touch
        </Button>
      </div>
    </div>
  );
}

export default HeroSection;
