#!/usr/bin/python3

import matplotlib.pyplot as plt


def main():
	# TODO: for each compression level in {0, 1, ..., 9}:
	#			fill real_time [ms]
	#			fill archive_size [bytes]
	level        = list(range(10))
	real_time    = [.712784635,1.249143104,.381370806,.902387035,.471783388,.656091280,.939214337,1.388082962,1.615415085,1.397006439]
	archive_size = [ 12206080, 5765351,5574195,5405653,5310629,5188651,5132846,5122181,5113410,5113141]

	archive_size = [it / 1024 for it in archive_size]

	# plot data
	fix, axs = plt.subplots(2)

	axs[0].plot(level, real_time,    'bo-')
	axs[1].plot(level, archive_size, 'ro-')

	axs[0].set_xticks(level)
	axs[1].set_xticks(level)

	axs[0].grid(True, which='both')
	axs[1].grid(True, which='both')

	axs[0].set_xlabel('Compression level')
	axs[1].set_xlabel('Compression level')

	axs[0].set_ylabel('Time [ms]')
	axs[1].set_ylabel('Archive size [byte]')

	plt.show()


if __name__ == '__main__':
	main()
