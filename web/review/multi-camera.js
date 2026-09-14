'use strict';
// The top camera owns the clock; wrist views follow without independent controls.
class MultiCamera {
  constructor(master, followers, onError) {
    this.master = master;
    this.followers = followers;
    this.onError = onError;
    this.generation = 0;
    for (const follower of followers) {
      follower.muted = true;
      follower.addEventListener('loadedmetadata', () => this.sync(true));
      follower.addEventListener('error', () => onError(new Error('腕相机无法加载：' + follower.getAttribute('src'))));
    }
    master.addEventListener('play', () => this.play());
    master.addEventListener('pause', () => { this.pause(); this.sync(true); });
    master.addEventListener('seeking', () => { this.pause(); this.sync(true); });
    master.addEventListener('seeked', () => { this.sync(true); if (!master.paused) this.play(); });
    master.addEventListener('ratechange', () => this.sync(false));
    master.addEventListener('waiting', () => this.pause());
    master.addEventListener('playing', () => this.play());
    master.addEventListener('ended', () => { this.pause(); this.sync(true); });
    const tick = () => { this.sync(false); master.requestVideoFrameCallback(tick); };
    master.requestVideoFrameCallback(tick);
  }
  load(episode) {
    this.generation++;
    this.pause();
    for (const [index, arm] of ['left_wrist', 'right_wrist'].entries()) {
      this.followers[index].src = `videos/${arm}/episode_${String(episode).padStart(3, '0')}.mp4`;
      this.followers[index].load();
    }
  }
  pause() { this.followers.forEach(video => video.pause()); }
  sync(force) {
    for (const video of this.followers) {
      video.playbackRate = this.master.playbackRate;
      if (video.readyState < 1) continue;
      if (force || Math.abs(video.currentTime - this.master.currentTime) > 1 / 30) {
        video.currentTime = Math.min(this.master.currentTime, Math.max(0, video.duration - 1 / 30));
      }
    }
  }
  play() {
    if (this.master.paused || this.master.seeking) return;
    this.sync(true);
    const generation = this.generation;
    for (const video of this.followers) {
      video.play().catch(error => {
        if (error.name === 'AbortError' && (generation !== this.generation || this.master.paused || this.master.seeking || video.paused)) return;
        this.master.pause();
        this.onError(error);
      });
    }
  }
}
