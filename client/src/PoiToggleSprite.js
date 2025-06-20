import React, { useState, useMemo } from 'react';
import './PoiToggleSprite.css';

export default function PoiToggleSprite({
  src = '/icons/eye_sprite.png',
  frameCount = 125,
  frameWidth = 350,    
  frameHeight = 350,   
  hoverFraction = 0.2,
  activeFraction = 0.5,
  displayWidth = 120,     
  displayHeight = 120
}) {
  const [active, setActive] = useState(false);
  const [hover,  setHover]  = useState(false);

  const frameIndex = useMemo(() => {
    const frac = active ? activeFraction : hover ? hoverFraction : 0;
    return Math.round((frameCount - 1) * frac);
  }, [active, hover, frameCount, hoverFraction, activeFraction]);

  const bgPosX = -frameIndex * displayWidth;
  const bgSizeX = frameCount * displayWidth;
  const bgSizeY = displayHeight;

  return (
    <div
      className="poi-toggle-sprite"
      style={{
        width:           `${displayWidth}px`,
        height:          `${displayHeight}px`,
        backgroundImage: `url(${src})`,
        backgroundRepeat:'no-repeat',
        backgroundPosition: `${bgPosX}px center`,
        backgroundSize:  `${bgSizeX}px ${bgSizeY}px`
      }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={() => setActive(a => !a)}
    />
  );
}
