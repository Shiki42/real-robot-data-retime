'use strict';
class IdleTimeline {
  constructor(root, seek) {
    this.root = root;
    this.seek = seek;
    this.tracks = [];
  }
  setEpisode(row, masks) {
    if (masks.output !== row.output || masks.frames !== row.frames || masks.source !== row.source) {
      throw new Error('Idle mask 与当前视频不匹配');
    }
    this.row = row;
    this.root.replaceChildren();
    this.tracks = ['left', 'right'].map((arm, index) => {
      const intervals = masks[arm];
      let last = 0, total = 0;
      for (const [start, end] of intervals) {
        if (!Number.isInteger(start) || !Number.isInteger(end) || start < last || start >= end || end > row.frames) {
          throw new Error('Idle mask 区间无效');
        }
        last = end;
        total += end - start;
      }
      const line = document.createElement('div');
      line.className = 'idle-row';
      const label = document.createElement('div');
      label.className = 'idle-label';
      label.textContent = `${index === 0 ? '左臂' : '右臂'} · 屏蔽 ${total} 帧 / ${(total / row.fps).toFixed(2)} 秒`;
      const status = document.createElement('span');
      status.className = 'idle-status';
      label.append(status);
      const track = document.createElement('div');
      track.className = 'idle-track';
      track.tabIndex = 0;
      track.setAttribute('role', 'slider');
      track.setAttribute('aria-label', `${index === 0 ? '左臂' : '右臂'} idle mask 时间轴`);
      track.setAttribute('aria-valuemin', '0');
      track.setAttribute('aria-valuemax', String(row.frames - 1));
      for (const [start, end] of intervals) {
        const span = document.createElement('span');
        span.className = 'idle-span';
        span.style.left = `${start / row.frames * 100}%`;
        span.style.width = `${(end - start) / row.frames * 100}%`;
        span.title = `屏蔽 action loss · 帧 ${start}–${end - 1} · ${(start / row.fps).toFixed(3)}–${(end / row.fps).toFixed(3)} 秒`;
        track.append(span);
      }
      const cursor = document.createElement('span');
      cursor.className = 'idle-cursor';
      track.append(cursor);
      track.onclick = event => {
        const bounds = track.getBoundingClientRect();
        this.seek(Math.min(row.frames - 1, Math.max(0, Math.floor((event.clientX - bounds.left) / bounds.width * row.frames))));
      };
      track.onkeydown = event => {
        const keys = {ArrowLeft: this.current - 1, ArrowRight: this.current + 1, Home: 0, End: row.frames - 1};
        if (Object.hasOwn(keys, event.key)) {
          event.preventDefault();
          event.stopPropagation();
          this.seek(Math.max(0, Math.min(row.frames - 1, keys[event.key])));
        }
      };
      line.append(label, track);
      this.root.append(line);
      return {intervals, status, cursor, track};
    });
    const scale = document.createElement('div');
    scale.className = 'idle-scale';
    for (let i = 0; i <= 4; i++) {
      const tick = document.createElement('span');
      tick.textContent = `${(row.frames / row.fps * i / 4).toFixed(2)}s`;
      scale.append(tick);
    }
    this.root.append(scale);
    this.update(0);
  }
  update(frame) {
    if (!this.row) return;
    this.current = Math.min(this.row.frames - 1, Math.max(0, frame));
    for (const {intervals, status, cursor, track} of this.tracks) {
      const idle = intervals.some(([start, end]) => start <= this.current && this.current < end);
      status.textContent = idle ? '当前：idle · loss=0' : '当前：保留监督 · loss=1';
      status.classList.toggle('masked', idle);
      cursor.style.left = `${(this.current + 0.5) / this.row.frames * 100}%`;
      track.setAttribute('aria-valuenow', String(this.current));
      track.setAttribute('aria-valuetext', `第 ${this.current} 帧，${idle ? '屏蔽' : '保留'} action loss`);
    }
  }
}
