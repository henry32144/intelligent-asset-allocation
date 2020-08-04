import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Typography from '@material-ui/core/Typography';
import IconButton from '@material-ui/core/IconButton';
import ShowChartIcon from '@material-ui/icons/ShowChart';
import BarChartIcon from '@material-ui/icons/BarChart';
import ListAltIcon from '@material-ui/icons/ListAlt';
import SearchIcon from '@material-ui/icons/Search';
import Tooltip from '@material-ui/core/Tooltip';
import { NEWS_SECTION, PERFORMANCE_SECTION, WEIGHT_SECTION } from '../Constants';

const useStyles = makeStyles((theme) => ({
  root: {
    marginLeft: "auto",
    display: 'inline-flex',
  },
  popover: {
    pointerEvents: 'none',
  },
  paper: {
    padding: theme.spacing(1),
  },
}));

export default function PortfolioDetailButtons(props) {
  const classes = useStyles();

  const newsButtonOnclick = () => {
    //console.log(PERFORMANCE_SECTION);
    props.setSectionCode(NEWS_SECTION);
  }

  const performanceButtonOnclick = () => {
    //console.log(PERFORMANCE_SECTION);
    props.setSectionCode(PERFORMANCE_SECTION);
  }

  const weightButtonOnclick = () => {
    //console.log(WEIGHT_SECTION);
    props.setSectionCode(WEIGHT_SECTION);
  }

  return (
    <div className={classes.root}>
      {/* {props.showSearchButton &&
        <IconButton
          edge="start"
          color="inherit"
          aria-label="expand search stock">
          <SearchIcon />
        </IconButton>
      } */}
      <Tooltip title="News">
        <IconButton
          edge="start"
          color="inherit"
          onClick={newsButtonOnclick}
          aria-label="show News">
          <ListAltIcon />
        </IconButton>
      </Tooltip>
      <Tooltip title="Performance">
        <IconButton
          edge="start"
          color="inherit"
          onClick={performanceButtonOnclick}
          aria-label="show performance">
          <ShowChartIcon />
        </IconButton>
      </Tooltip>
      <Tooltip title="Weight">
        <IconButton
          edge="start"
          color="inherit"
          onClick={weightButtonOnclick}
          aria-label="asset weight">
          <BarChartIcon />
        </IconButton>
      </Tooltip>
    </div>
  );
}
